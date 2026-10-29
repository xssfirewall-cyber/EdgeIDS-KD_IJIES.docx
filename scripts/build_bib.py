"""Builds paper/refs.bib from registry metadata (Crossref / DataCite), not from
memory. Every entry's DOI is resolved; title, authors, venue, volume, issue,
pages and year come from the registry record. Only Holm (1979) has no DOI in
any registry and is written from the journal record.

    python scripts/build_bib.py   -> paper/refs.bib, paper/refs_audit.csv
"""
import csv
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from difflib import SequenceMatcher
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

UA = {"User-Agent": "EdgeIDS-KD-refcheck/1.0 (academic reference verification)"}

# key -> (DOI, title as expected; used to verify the registry record,
#         optional note on the peer-reviewed venue of an arXiv preprint)
REFS = {
    "hinton2015distilling": ("10.48550/arXiv.1503.02531", "Distilling the Knowledge in a Neural Network", None),
    "menon2021logit": ("10.48550/arXiv.2007.07314", "Long-tail learning via logit adjustment", "Published at ICLR 2021"),
    "cui2019class": ("10.1109/CVPR.2019.00949", "Class-Balanced Loss Based on Effective Number of Samples", None),
    "hooker2019forget": ("10.48550/arXiv.1911.05248", "What Do Compressed Deep Neural Networks Forget?", None),
    "mirzadeh2020improved": ("10.1609/aaai.v34i04.5963", "Improved Knowledge Distillation via Teacher Assistant", None),
    "jacob2018quantization": ("10.1109/CVPR.2018.00286", "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference", None),
    "gorishniy2021revisiting": ("10.48550/arXiv.2106.11959", "Revisiting Deep Learning Models for Tabular Data", "Published at NeurIPS 2021"),
    "meidan2018nbaiot": ("10.1109/MPRV.2018.03367731", "N-BaIoT Network-Based Detection of IoT Botnet Attacks Using Deep Autoencoders", None),
    "moustafa2015unsw": ("10.1109/MilCIS.2015.7348942", "UNSW-NB15: a comprehensive data set for network intrusion detection systems (UNSW-NB15 network data set)", None),
    "alsaedi2020toniot": ("10.1109/ACCESS.2020.3022862", "TON_IoT Telemetry Dataset: A New Generation Dataset of IoT and IIoT for Data-Driven Intrusion Detection Systems", None),
    "neto2023ciciot": ("10.3390/s23135941", "CICIoT2023: A Real-Time Dataset and Benchmark for Large-Scale Attacks in IoT Environment", None),
    "ferrag2022edgeiiot": ("10.1109/ACCESS.2022.3165809", "Edge-IIoTset: A New Comprehensive Realistic Cyber Security Dataset of IoT and IIoT Applications for Centralized and Federated Learning", None),
    "arp2022dos": ("10.48550/arXiv.2010.09470", "Dos and Don'ts of Machine Learning in Computer Security", "Published at USENIX Security 2022"),
    "engelen2021troubleshooting": ("10.1109/SPW53761.2021.00009", "Troubleshooting an Intrusion Detection Dataset: the CICIDS2017 Case Study", None),
    "sommer2010outside": ("10.1109/SP.2010.25", "Outside the Closed World: On Using Machine Learning for Network Intrusion Detection", None),
    "zhang2023balanced": ("10.48550/arXiv.2104.10510", "Balanced Knowledge Distillation for Long-tailed Learning", None),
    "benaddi2025lightweight": ("10.48550/arXiv.2512.19488", "Lightweight Intrusion Detection in IoT via SHAP-Guided Feature Pruning and Knowledge-Distilled Kronecker Networks", None),
    "yagiz2025lens": ("10.48550/arXiv.2501.00790", "LENS-XAI: Redefining Lightweight and Explainable Network Security through Knowledge Distillation and Variational Autoencoders for Scalable Intrusion Detection in Cybersecurity", None),
    "yang2023lightweight": ("10.48550/arXiv.2307.10191", "A Lightweight Approach for Network Intrusion Detection based on Self-Knowledge Distillation", "Accepted at IEEE ICC 2023"),
    "alikhani2026dkd": ("10.48550/arXiv.2603.03486", "DKD-KAN: A Lightweight knowledge-distilled KAN intrusion detection framework, based on MLP and KAN", None),
    "debernardis2026lightweight": ("10.3390/electronics15132869", "Lightweight Machine Learning Intrusion Detection for IoT/IIoT Networks: Quantisation Strategies and Physical Deployment on Resource-Constrained Microcontrollers", None),
    "park2026reliable": ("10.48550/arXiv.2609.27202", "Reliable Federated TinyML Deployment for IoT Security", None),
    "zarpelao2017survey": ("10.1016/j.jnca.2017.02.009", "A survey of intrusion detection in Internet of Things", None),
    "chaabouni2019network": ("10.1109/COMST.2019.2896380", "Network Intrusion Detection for IoT Security Based on Learning Techniques", None),
    "kolias2017ddos": ("10.1109/MC.2017.201", "DDoS in the IoT: Mirai and Other Botnets", None),
    "sharafaldin2018toward": ("10.5220/0006639801080116", "Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization", None),
    "koroniotis2019towards": ("10.1016/j.future.2019.05.041", "Towards the development of realistic botnet dataset in the Internet of Things for network forensic analytics: Bot-IoT dataset", None),
    "mirsky2018kitsune": ("10.14722/ndss.2018.23204", "Kitsune: An Ensemble of Autoencoders for Online Network Intrusion Detection", None),
    "ferrag2020deep": ("10.1016/j.jisa.2019.102419", "Deep learning for cyber security intrusion detection: Approaches, datasets, and comparative study", None),
    "gou2021knowledge": ("10.1007/s11263-021-01453-z", "Knowledge Distillation: A Survey", None),
    "zhao2022decoupled": ("10.1109/CVPR52688.2022.01165", "Decoupled Knowledge Distillation", None),
    "he2009learning": ("10.1109/TKDE.2008.239", "Learning from Imbalanced Data", None),
    "johnson2019survey": ("10.1186/s40537-019-0192-5", "Survey on deep learning with class imbalance", None),
    "lin2020focal": ("10.1109/TPAMI.2018.2858826", "Focal Loss for Dense Object Detection", None),
    "chawla2002smote": ("10.1613/jair.953", "SMOTE: Synthetic Minority Over-sampling Technique", None),
    "kapoor2023leakage": ("10.1016/j.patter.2023.100804", "Leakage and the reproducibility crisis in machine-learning-based science", None),
    "kaufman2012leakage": ("10.1145/2382577.2382579", "Leakage in data mining: Formulation, detection, and avoidance", None),
    "chen2016xgboost": ("10.1145/2939672.2939785", "XGBoost: A Scalable Tree Boosting System", None),
    "hochreiter1997long": ("10.1162/neco.1997.9.8.1735", "Long Short-Term Memory", None),
    "banbury2021mlperf": ("10.48550/arXiv.2106.07597", "MLPerf Tiny Benchmark", None),
    "wilcoxon1945individual": ("10.2307/3001968", "Individual Comparisons by Ranking Methods", None),
    "efron1979bootstrap": ("10.1214/aos/1176344552", "Bootstrap Methods: Another Look at the Jackknife", None),
}

# No DOI exists in Crossref or DataCite (checked 2026-10-14).
HOLM = """@article{holm1979simple,
  author  = {Holm, Sture},
  title   = {A Simple Sequentially Rejective Multiple Test Procedure},
  journal = {Scandinavian Journal of Statistics},
  volume  = {6},
  number  = {2},
  pages   = {65--70},
  year    = {1979}
}
"""


def get(url):
    last = None
    for _ in range(3):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30) as r:
                return json.load(r)
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(1.5)
    raise last


def norm(s):
    return re.sub(r"[^a-z0-9]+", " ", re.sub(r"<[^>]+>", " ", s or "").lower()).strip()


def tex(s):
    s = re.sub(r"<[^>]+>", "", s or "")
    # Unescape HTML first, then escape each "&" exactly once.
    s = s.replace("—", "--").replace("–", "--").replace("&amp;", "&").replace("&", "\\&")
    s = s.replace("_", "\\_").replace("%", "\\%").replace("#", "\\#")
    return s


PROPER = ("Internet of Things", "Mirai", "Kronecker", "Kolmogorov", "Arnold", "Jackknife")


def protect(title):
    # keep acronyms and capitalised technical words as written
    title = re.sub(r"\b([A-Za-z]*[A-Z][A-Za-z0-9\-]*[A-Z0-9][A-Za-z0-9\-]*)\b", r"{\1}", title)
    for p in PROPER:   # proper nouns with a single capital, which the style would lowercase
        title = re.sub(rf"(?i)\b{re.escape(p)}\b", "{" + p + "}", title)
    return title


def crossref(doi):
    m = get("https://api.crossref.org/works/" + urllib.parse.quote(doi))["message"]
    title = (m.get("title") or [""])[0]
    if m.get("subtitle"):
        title = f"{title}: {m['subtitle'][0]}"
    year = (m.get("published-print") or m.get("published-online") or m.get("issued"))["date-parts"][0][0]
    authors = [f"{a.get('family', '')}, {a.get('given', '')}".strip(", ") for a in m.get("author", [])]
    typ = m.get("type")
    entry = {"title": title, "year": str(year), "author": " and ".join(authors),
             "volume": m.get("volume"), "number": m.get("issue"),
             "pages": (m.get("page") or "").replace("-", "--") or None}
    venue = (m.get("container-title") or [""])[0]
    if typ == "journal-article":
        entry["journal"] = venue
        kind = "article"
    else:
        entry["booktitle"] = venue
        kind = "inproceedings"
    return kind, entry


def datacite(doi):
    a = get("https://api.datacite.org/dois/" + urllib.parse.quote(doi))["data"]["attributes"]
    authors = []
    for c in a.get("creators", []):
        if c.get("familyName"):
            authors.append(f"{c['familyName']}, {c.get('givenName', '')}".strip(", "))
        else:
            authors.append(c.get("name", ""))
    arxiv_id = doi.split("arXiv.")[-1]
    return "article", {"title": a["titles"][0]["title"], "year": str(a.get("publicationYear")),
                       "author": " and ".join(authors), "journal": f"arXiv preprint arXiv:{arxiv_id}"}


def main():
    out, audit = [], []
    for key, (doi, expected, note) in REFS.items():
        kind, e = datacite(doi) if doi.lower().startswith("10.48550/") else crossref(doi)
        sim = SequenceMatcher(None, norm(expected), norm(e["title"])).ratio()
        if sim < 0.85:
            raise SystemExit(f"title mismatch for {key}: registry '{e['title']}' (sim {sim:.2f})")
        e["doi"] = doi
        # IEEEtranN does not print the doi field, so the DOI also goes into the
        # note, in IEEE's "doi: ..." form, as a link.
        doi_text = f"doi: \\href{{https://doi.org/{doi}}}{{{doi}}}"
        e["note"] = f"{note}. {doi_text}" if note else doi_text
        fields = [f"  {k:<7} = {{{protect(tex(v)) if k == 'title' else (v if k == 'note' else tex(v))}}}"
                  for k, v in
                  (("author", e.get("author")), ("title", e["title"]), ("journal", e.get("journal")),
                   ("booktitle", e.get("booktitle")), ("volume", e.get("volume")), ("number", e.get("number")),
                   ("pages", e.get("pages")), ("year", e["year"]), ("doi", doi), ("note", e.get("note")))
                  if v]
        # DOIs must not be TeX-escaped
        fields = [f"  doi     = {{{doi}}}" if f.lstrip().startswith("doi") else f for f in fields]
        out.append(f"@{kind}{{{key},\n" + ",\n".join(fields) + "\n}\n")
        audit.append({"key": key, "doi": doi, "registry": "DataCite" if "48550" in doi else "Crossref",
                      "title_similarity": round(sim, 3), "registry_title": e["title"], "year": e["year"]})
        print(f"{sim:.2f} {key:<28} {doi}")
        time.sleep(0.25)
    header = ("% Generated by scripts/build_bib.py from Crossref/DataCite records (not from\n"
              "% memory). Every DOI resolves and its registry title matches the entry\n"
              "% (see refs_audit.csv). holm1979simple has no DOI in any registry.\n\n")
    paper = config.ROOT / "paper"
    (paper / "refs.bib").write_text(header + "\n".join(out) + "\n" + HOLM, encoding="utf-8")
    with open(paper / "refs_audit.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(audit[0]))
        w.writeheader()
        w.writerows(audit)
    print(f"\n{len(out)} entries with DOI + Holm (no DOI) = {len(out) + 1}")


if __name__ == "__main__":
    main()
