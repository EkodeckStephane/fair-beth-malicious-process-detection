#!/usr/bin/env python3
"""
Génération du mapping MITRE ATT&CK depuis les eventIds du dataset BETH.
Le mapping est enrichi par heuristique sur les IDs BETH et les eventName observés.
"""
import os
import json
import pandas as pd
from config import MITRE_MAP_PATH, OUTPUT_DIR, DATA_DIR
from utils import ensure_dir, save_json

# Mapping enrichi basé sur l'analyse des eventIds BETH et leur sémantique Sysmon-like.
# Les IDs 1-25 correspondent aux événements de base ; les 1000+ à Windows Security.
MITRE_MAPPING_RAW = {
    # === Core Sysmon-like events (IDs observés dans BETH) ===
    "1":   ["T1059", "T1106"],          # Process Create -> Execution
    "2":   ["T1562"],                   # Process Terminated -> Defense Evasion
    "3":   ["T1041", "T1071", "T1021"], # Network Connect -> C2 / Exfiltration
    "5":   ["T1486", "T1027"],          # File Create -> Encryption / Obfuscation
    "6":   ["T1112"],                   # Registry Event -> Modify Registry
    "7":   ["T1055", "T1574"],           # Image Load -> Process Injection
    "8":   ["T1055"],                   # Create Remote Thread -> Injection
    "9":   ["T1005", "T1567"],          # Raw Access Read -> Collection / Exfil
    "10":  ["T1055", "T1003"],          # Process Access -> Injection / Credential Access
    "11":  ["T1486", "T1027", "T1041"], # File Create (alt) -> Encryption
    "12":  ["T1112"],                   # Registry Event (alt)
    "13":  ["T1112"],                   # Registry Value Set
    "14":  ["T1567"],                   # File Stream Created
    "15":  ["T1027"],                   # File Create Stream Hash -> Obfuscation
    "17":  ["T1021"],                   # Pipe Created -> Remote Services
    "18":  ["T1021"],                   # Pipe Connected
    "22":  ["T1071"],                   # DNS Query -> Application Layer Protocol
    "23":  ["T1485", "T1070"],          # File Delete Archived -> Impact / Indicator Removal
    "25":  ["T1562"],                   # Process Tampering -> Defense Evasion

    # === BETH-specific IDs (heuristic mapping from BETH eventId ranges) ===
    "21":  ["T1071"],                   # Likely DNS/Network event
    "33":  ["T1070"],                   # Indicator Removal
    "41":  ["T1055"],                   # Process Injection
    "42":  ["T1055"],                   # Process Injection
    "43":  ["T1070"],                   # Indicator Removal
    "49":  ["T1562"],                   # Defense Evasion
    "50":  ["T1562"],                   # Defense Evasion
    "51":  ["T1071"],                   # Network / C2
    "56":  ["T1027"],                   # Obfuscated Files
    "59":  ["T1112"],                   # Registry
    "62":  ["T1059"],                   # Command Execution
    "87":  ["T1041"],                   # Exfiltration Over C2
    "88":  ["T1041"],                   # Exfiltration Over C2
    "90":  ["T1071"],                   # Application Layer Protocol
    "91":  ["T1071"],                   # Application Layer Protocol
    "92":  ["T1499"],                   # Unknown / Error -> Anomaly
    "94":  ["T1055"],                   # Process Injection
    "105": ["T1486"],                   # Data Encrypted for Impact
    "106": ["T1486"],                   # Data Encrypted for Impact
    "113": ["T1005"],                   # Data from Local System
    "114": ["T1005"],                   # Data from Local System
    "122": ["T1562"],                   # Impair Defenses
    "123": ["T1070"],                   # Indicator Removal
    "133": ["T1027"],                   # Obfuscated Files
    "157": ["T1112"],                   # Modify Registry
    "165": ["T1055"],                   # Process Injection
    "166": ["T1055"],                   # Process Injection
    "217": ["T1071"],                   # Application Layer Protocol
    "257": ["T1562"],                   # Impair Defenses
    "260": ["T1059"],                   # Command Execution
    "263": ["T1070"],                   # Indicator Removal
    "268": ["T1005"],                   # Data Collection
    "269": ["T1070"],                   # Indicator Removal
    "288": ["T1041"],                   # Exfiltration Over C2
    "292": ["T1055"],                   # Process Injection
    "319": ["T1112"],                   # Modify Registry
    "321": ["T1486"],                   # Data Encrypted for Impact

    # === Windows Security events (1000+) ===
    "1003": ["T1070"],                  # Log cleared / deletion
    "1004": ["T1070"],                  # Log cleared
    "1005": ["T1005"],                  # Data access
    "1006": ["T1499"],                  # Error / Anomaly
    "1009": ["T1562"],                  # Tampering
    "1010": ["T1070"],                  # Log deletion

    # === Windows Security standard IDs ===
    "4688": ["T1059", "T1106"],         # Process Creation (Security)
    "4689": ["T1562"],                  # Process Termination
    "5156": ["T1021", "T1041"],         # WFP Network Connection
}

def generate_mitre_map(event_ids_present=None):
    if event_ids_present is None:
        return dict(MITRE_MAPPING_RAW)
    present = set(str(int(e)) for e in event_ids_present)
    filtered = {}
    for k, v in MITRE_MAPPING_RAW.items():
        if k in present:
            filtered[k] = v
    for eid in present:
        if eid not in filtered:
            filtered[eid] = ["T1499"]
    return filtered

def main():
    ensure_dir(OUTPUT_DIR)
    # Optionnel : lecture des eventName réels pour afficher un aperçu
    if os.path.isdir(DATA_DIR):
        all_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv") and not f.endswith("-dns.csv")]
        if all_files:
            sample = pd.read_csv(os.path.join(DATA_DIR, all_files[0]), nrows=1000, low_memory=False)
            if "eventId" in sample.columns and "eventName" in sample.columns:
                mapping = sample.groupby("eventId")["eventName"].first().to_dict()
                print("[*] EventIds observés dans le dataset (aperçu) :")
                for eid, name in sorted(mapping.items()):
                    eid_str = str(int(eid))
                    techs = MITRE_MAPPING_RAW.get(eid_str, ["T1499"])
                    print(f"    eventId={eid_str:>5s} | eventName={name:30s} | MITRE={', '.join(techs)}")

    save_json(MITRE_MAPPING_RAW, MITRE_MAP_PATH)
    print(f"[+] MITRE mapping enrichi sauvegardé : {MITRE_MAP_PATH}")
    print(f"    {len(MITRE_MAPPING_RAW)} eventIds mappés.")

if __name__ == "__main__":
    main()
