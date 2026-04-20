#!/usr/bin/env python3
"""
Download Molecule Data
======================

Unified script for downloading small molecule data from multiple sources:
1. ChEMBL - Bioactivity database with drug-like molecules
2. PubChem - Large chemical compound database
3. DrugBank - Approved drugs information

This script provides molecule data for protein-molecule interaction models.

Usage:
    python download_mol.py --source chembl --target-organism "Homo sapiens"
    python download_mol.py --source chembl --max-molecules 10000
    python download_mol.py --smiles-file molecules.smi
    python download_mol.py --chembl-ids CHEMBL25,CHEMBL192

Data Sources:
    - ChEMBL: Bioactivity data for drug-like molecules
    - PubChem: Chemical compound structures
    - DrugBank: Approved drug information
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import requests
from tqdm import tqdm

# Ensure unbuffered output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

# =============================================================================
# API ENDPOINTS
# =============================================================================

# ChEMBL
CHEMBL_BASE_URL = "https://www.ebi.ac.uk/chembl/api/data"
CHEMBL_MOLECULE_URL = f"{CHEMBL_BASE_URL}/molecule.json"
CHEMBL_ACTIVITY_URL = f"{CHEMBL_BASE_URL}/activity.json"
CHEMBL_TARGET_URL = f"{CHEMBL_BASE_URL}/target.json"
CHEMBL_MECHANISM_URL = "https://www.ebi.ac.uk/chembl/api/rest/mechanism.json"

# PubChem
PUBCHEM_BASE_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"

# =============================================================================
# SMILES UTILITIES
# =============================================================================


def is_valid_smiles(smiles: str) -> bool:
    """
    Basic validation of SMILES string

    Args:
        smiles: SMILES string

    Returns:
        True if SMILES looks valid
    """
    if not smiles or len(smiles) < 2:
        return False

    # Check for basic SMILES characters
    valid_chars = set(
        "CNOSPFClBrI[]()=#@+\\/-0123456789cnospabcdefghijklmnopqrstuvwxyz."
    )
    return all(c in valid_chars for c in smiles)


def canonicalize_smiles(smiles: str) -> str:
    """
    Attempt to canonicalize SMILES (basic cleanup)

    Args:
        smiles: Input SMILES string

    Returns:
        Cleaned SMILES string
    """
    # Basic cleanup - remove whitespace
    return smiles.strip()


# =============================================================================
# CHEMBL FUNCTIONS
# =============================================================================


def fetch_chembl_molecules(
    max_molecules: int = 10000,
    include_activities: bool = False,
    target_organism: Optional[str] = None,
) -> List[Dict]:
    """
    Fetch molecules from ChEMBL database

    Args:
        max_molecules: Maximum number of molecules to fetch
        include_activities: Include bioactivity data
        target_organism: Filter by target organism

    Returns:
        List of molecule dictionaries
    """
    print("Fetching molecules from ChEMBL...")

    molecules = []
    offset = 0
    limit = 1000

    # Build query parameters
    params = {
        "limit": limit,
        "offset": offset,
        "molecule_type": "Small molecule",
    }

    with tqdm(total=max_molecules, desc="Fetching molecules") as pbar:
        while len(molecules) < max_molecules:
            params["offset"] = offset

            try:
                response = requests.get(CHEMBL_MOLECULE_URL, params=params, timeout=60)
                response.raise_for_status()
                data = response.json()

                batch = data.get("molecules", [])
                if not batch:
                    break

                for mol in batch:
                    if len(molecules) >= max_molecules:
                        break

                    chembl_id = mol.get("molecule_chembl_id")
                    structures = mol.get("molecule_structures")

                    if not chembl_id or not structures:
                        continue

                    smiles = structures.get("canonical_smiles")
                    if not smiles or not is_valid_smiles(smiles):
                        continue

                    molecule_data = {
                        "chembl_id": chembl_id,
                        "smiles": smiles,
                        "name": mol.get("pref_name", ""),
                        "max_phase": mol.get("max_phase", 0),
                        "molecule_type": mol.get("molecule_type", ""),
                        "molecular_weight": mol.get("molecule_properties", {}).get(
                            "full_mwt"
                        ),
                        "alogp": mol.get("molecule_properties", {}).get("alogp"),
                        "hba": mol.get("molecule_properties", {}).get("hba"),
                        "hbd": mol.get("molecule_properties", {}).get("hbd"),
                        "psa": mol.get("molecule_properties", {}).get("psa"),
                        "num_ro5_violations": mol.get("molecule_properties", {}).get(
                            "num_ro5_violations"
                        ),
                    }

                    molecules.append(molecule_data)
                    pbar.update(1)

                offset += limit
                time.sleep(0.3)  # Rate limiting

            except requests.RequestException as e:
                print(f"\n⚠️ Error at offset {offset}: {e}")
                time.sleep(1)
                continue

    print(f"\n✓ Fetched {len(molecules):,} molecules from ChEMBL")
    return molecules


def fetch_chembl_activities(
    target_organism: str = "Homo sapiens",
    activity_types: Optional[List[str]] = None,
    max_activities: int = 50000,
    min_confidence: int = 8,
) -> List[Dict]:
    """
    Fetch bioactivity data from ChEMBL

    Args:
        target_organism: Target organism filter
        activity_types: List of activity types (IC50, Ki, Kd, EC50)
        max_activities: Maximum number of activities to fetch
        min_confidence: Minimum target confidence score

    Returns:
        List of activity records
    """
    print(f"\nFetching ChEMBL activities for {target_organism}...")

    if activity_types is None:
        activity_types = ["IC50", "Ki", "Kd", "EC50"]

    activities = []
    offset = 0
    limit = 1000

    params = {
        "target_organism": target_organism,
        "target_type": "SINGLE PROTEIN",
        "assay_type": "B",  # Binding assays
        "standard_type__in": ",".join(activity_types),
        "standard_relation": "=",
        "target_confidence_score__gte": min_confidence,
        "limit": limit,
    }

    with tqdm(total=max_activities, desc="Fetching activities") as pbar:
        while len(activities) < max_activities:
            params["offset"] = offset

            try:
                response = requests.get(CHEMBL_ACTIVITY_URL, params=params, timeout=60)
                response.raise_for_status()
                data = response.json()

                batch = data.get("activities", [])
                if not batch:
                    break

                for act in batch:
                    if len(activities) >= max_activities:
                        break

                    activities.append(
                        {
                            "molecule_chembl_id": act.get("molecule_chembl_id"),
                            "target_chembl_id": act.get("target_chembl_id"),
                            "standard_type": act.get("standard_type"),
                            "standard_value": act.get("standard_value"),
                            "standard_units": act.get("standard_units"),
                            "pchembl_value": act.get("pchembl_value"),
                            "assay_chembl_id": act.get("assay_chembl_id"),
                        }
                    )
                    pbar.update(1)

                offset += limit
                time.sleep(0.3)

            except requests.RequestException as e:
                print(f"\n⚠️ Error at offset {offset}: {e}")
                time.sleep(1)
                continue

    print(f"✓ Fetched {len(activities):,} activities")
    return activities


def fetch_molecules_for_activities(activities: List[Dict]) -> Dict[str, str]:
    """
    Fetch SMILES for molecules in activity data

    Args:
        activities: List of activity records

    Returns:
        Dictionary mapping ChEMBL IDs to SMILES
    """
    # Get unique molecule IDs
    molecule_ids = list(
        set(
            act["molecule_chembl_id"]
            for act in activities
            if act.get("molecule_chembl_id")
        )
    )
    print(f"\nFetching SMILES for {len(molecule_ids):,} molecules...")

    smiles_dict: Dict[str, str] = {}
    batch_size = 50

    for i in tqdm(range(0, len(molecule_ids), batch_size), desc="Fetching SMILES"):
        batch = molecule_ids[i : i + batch_size]
        batch_str = ",".join(batch)

        params = {
            "molecule_chembl_id__in": batch_str,
            "limit": batch_size,
        }

        try:
            response = requests.get(CHEMBL_MOLECULE_URL, params=params, timeout=60)
            response.raise_for_status()
            data = response.json()

            for mol in data.get("molecules", []):
                chembl_id = mol.get("molecule_chembl_id")
                structures = mol.get("molecule_structures")
                if structures:
                    smiles = structures.get("canonical_smiles")
                    if smiles and chembl_id:
                        smiles_dict[chembl_id] = smiles

            time.sleep(0.2)

        except requests.RequestException as e:
            print(f"\n⚠️ Error: {e}")
            continue

    print(f"✓ Retrieved SMILES for {len(smiles_dict):,} molecules")
    return smiles_dict


def fetch_target_info(target_ids: List[str]) -> Dict[str, Dict]:
    """
    Fetch target information from ChEMBL

    Args:
        target_ids: List of ChEMBL target IDs

    Returns:
        Dictionary mapping target IDs to target info
    """
    print(f"\nFetching info for {len(target_ids):,} targets...")

    targets: Dict[str, Dict] = {}
    batch_size = 50

    for i in tqdm(range(0, len(target_ids), batch_size), desc="Fetching targets"):
        batch = target_ids[i : i + batch_size]
        batch_str = ",".join(batch)

        params = {
            "target_chembl_id__in": batch_str,
            "limit": batch_size,
        }

        try:
            response = requests.get(CHEMBL_TARGET_URL, params=params, timeout=60)
            response.raise_for_status()
            data = response.json()

            for target in data.get("targets", []):
                target_id = target.get("target_chembl_id")
                components = target.get("target_components", [])

                if target_id and components:
                    component = components[0]
                    accession = component.get("accession")

                    targets[target_id] = {
                        "chembl_id": target_id,
                        "uniprot_id": accession,
                        "pref_name": target.get("pref_name", ""),
                        "organism": target.get("organism", ""),
                        "target_type": target.get("target_type", ""),
                    }

            time.sleep(0.2)

        except requests.RequestException as e:
            print(f"\n⚠️ Error: {e}")
            continue

    print(f"✓ Retrieved info for {len(targets):,} targets")
    return targets


def fetch_mechanism_data(molecule_ids: List[str]) -> Dict[str, str]:
    """
    Fetch mechanism-of-action (action_type) for molecules from ChEMBL.

    Hits the /mechanism/ endpoint in batches of 50. When a molecule has
    multiple mechanisms (different targets), the first returned is used.

    Args:
        molecule_ids: List of ChEMBL molecule IDs

    Returns:
        Dict mapping molecule_chembl_id -> action_type string
        e.g. {"CHEMBL324340": "INHIBITOR", "CHEMBL109600": "ANTAGONIST"}
    """
    print(f"\nFetching mechanism-of-action for {len(molecule_ids):,} molecules...")

    result: Dict[str, str] = {}
    batch_size = 50

    for i in tqdm(range(0, len(molecule_ids), batch_size), desc="Fetching mechanisms"):
        batch = molecule_ids[i : i + batch_size]
        batch_str = ",".join(batch)

        params = {
            "molecule_chembl_id__in": batch_str,
            "limit": batch_size,
        }

        try:
            response = requests.get(CHEMBL_MECHANISM_URL, params=params, timeout=60)
            response.raise_for_status()
            data = response.json()

            for mechanism in data.get("mechanisms", []):
                mol_id = mechanism.get("molecule_chembl_id")
                action_type = mechanism.get("action_type")

                if not mol_id or not action_type:
                    continue

                if mol_id not in result:
                    result[mol_id] = action_type

            time.sleep(0.2)

        except requests.RequestException as e:
            print(f"\n⚠️ Warning: {e}")
            continue

    print(f"✓ Retrieved mechanisms for {len(result):,} molecules")
    return result


def fetch_chembl_by_ids(chembl_ids: List[str]) -> List[Dict]:
    """
    Fetch specific molecules by ChEMBL IDs

    Args:
        chembl_ids: List of ChEMBL IDs

    Returns:
        List of molecule dictionaries
    """
    print(f"Fetching {len(chembl_ids)} molecules by ID...")

    molecules = []
    batch_size = 50

    for i in tqdm(range(0, len(chembl_ids), batch_size), desc="Fetching"):
        batch = chembl_ids[i : i + batch_size]
        batch_str = ",".join(batch)

        params = {
            "molecule_chembl_id__in": batch_str,
            "limit": batch_size,
        }

        try:
            response = requests.get(CHEMBL_MOLECULE_URL, params=params, timeout=60)
            response.raise_for_status()
            data = response.json()

            for mol in data.get("molecules", []):
                chembl_id = mol.get("molecule_chembl_id")
                structures = mol.get("molecule_structures")

                if chembl_id and structures:
                    smiles = structures.get("canonical_smiles")
                    if smiles:
                        molecules.append(
                            {
                                "chembl_id": chembl_id,
                                "smiles": smiles,
                                "name": mol.get("pref_name", ""),
                                "max_phase": mol.get("max_phase", 0),
                            }
                        )

            time.sleep(0.2)

        except requests.RequestException as e:
            print(f"\n⚠️ Error: {e}")
            continue

    print(f"✓ Fetched {len(molecules)} molecules")
    return molecules


# =============================================================================
# PUBCHEM FUNCTIONS
# =============================================================================


def fetch_pubchem_by_smiles(smiles_list: List[str]) -> List[Dict]:
    """
    Fetch compound info from PubChem by SMILES

    Args:
        smiles_list: List of SMILES strings

    Returns:
        List of compound dictionaries
    """
    print(f"Fetching {len(smiles_list)} compounds from PubChem...")

    compounds = []

    for smiles in tqdm(smiles_list, desc="Fetching from PubChem"):
        url = f"{PUBCHEM_BASE_URL}/compound/smiles/{smiles}/JSON"

        try:
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                data = response.json()
                pc_compounds = data.get("PC_Compounds", [])

                if pc_compounds:
                    pc = pc_compounds[0]
                    cid = pc.get("id", {}).get("id", {}).get("cid")

                    if cid:
                        compounds.append(
                            {
                                "pubchem_cid": cid,
                                "smiles": smiles,
                            }
                        )

            time.sleep(0.2)  # Rate limiting

        except requests.RequestException:
            continue

    print(f"✓ Retrieved {len(compounds)} compounds from PubChem")
    return compounds


# =============================================================================
# FILE I/O
# =============================================================================


def load_smiles_file(filepath: Path) -> List[Dict]:
    """
    Load molecules from SMILES file

    Args:
        filepath: Path to SMILES file (.smi)

    Returns:
        List of molecule dictionaries
    """
    print(f"Loading SMILES from {filepath}...")

    molecules = []

    with open(filepath, "r") as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                smiles = parts[0]
                mol_id = parts[1] if len(parts) > 1 else f"mol_{len(molecules)}"

                if is_valid_smiles(smiles):
                    molecules.append(
                        {
                            "mol_id": mol_id,
                            "smiles": smiles,
                        }
                    )

    print(f"✓ Loaded {len(molecules)} molecules")
    return molecules


def save_molecules(
    molecules: List[Dict],
    output_dir: Path,
    activities: Optional[List[Dict]] = None,
    targets: Optional[Dict[str, Dict]] = None,
) -> None:
    """
    Save molecule data to files

    Args:
        molecules: List of molecule dictionaries
        output_dir: Output directory
        activities: Optional activity data
        targets: Optional target data
    """
    if not molecules:
        print("No molecules to save")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save SMILES file
    smi_file = output_dir / "molecules.smi"
    with open(smi_file, "w") as f:
        for mol in molecules:
            mol_id = mol.get("chembl_id") or mol.get("mol_id") or mol.get("pubchem_cid")
            smiles = mol.get("smiles", "")
            if smiles:
                f.write(f"{smiles}\t{mol_id}\n")
    print(f"✓ Saved SMILES: {smi_file}")

    # Save JSON with full data
    json_file = output_dir / "molecules.json"
    with open(json_file, "w") as f:
        json.dump(molecules, f, indent=2)
    print(f"✓ Saved JSON: {json_file}")

    # Save TSV summary
    tsv_file = output_dir / "molecules.tsv"
    with open(tsv_file, "w") as f:
        f.write("mol_id\tsmiles\tname\tmax_phase\tmolecular_weight\n")
        for mol in molecules:
            mol_id = mol.get("chembl_id") or mol.get("mol_id") or ""
            f.write(
                f"{mol_id}\t{mol.get('smiles', '')}\t{mol.get('name', '')}\t"
                f"{mol.get('max_phase', '')}\t{mol.get('molecular_weight', '')}\n"
            )
    print(f"✓ Saved TSV: {tsv_file}")

    # Save activities if provided
    if activities:
        act_file = output_dir / "activities.tsv"
        with open(act_file, "w") as f:
            f.write(
                "molecule_id\ttarget_id\tstandard_type\tstandard_value\t"
                "standard_units\tpchembl_value\n"
            )
            for act in activities:
                f.write(
                    f"{act.get('molecule_chembl_id', '')}\t"
                    f"{act.get('target_chembl_id', '')}\t"
                    f"{act.get('standard_type', '')}\t"
                    f"{act.get('standard_value', '')}\t"
                    f"{act.get('standard_units', '')}\t"
                    f"{act.get('pchembl_value', '')}\n"
                )
        print(f"✓ Saved activities: {act_file}")

    # Save targets if provided
    if targets:
        target_file = output_dir / "targets.tsv"
        with open(target_file, "w") as f:
            f.write("target_id\tuniprot_id\tpref_name\torganism\ttarget_type\n")
            for target_id, info in targets.items():
                f.write(
                    f"{target_id}\t{info.get('uniprot_id', '')}\t"
                    f"{info.get('pref_name', '')}\t{info.get('organism', '')}\t"
                    f"{info.get('target_type', '')}\n"
                )
        print(f"✓ Saved targets: {target_file}")


# =============================================================================
# METADATA AND SUMMARY
# =============================================================================


def save_metadata(
    output_dir: Path,
    source: str,
    molecule_count: int,
    activity_count: int = 0,
    target_organism: Optional[str] = None,
) -> None:
    """Save download metadata"""
    output_file = output_dir / "download_info.json"

    info = {
        "source": source,
        "molecule_count": molecule_count,
        "activity_count": activity_count,
        "target_organism": target_organism,
        "download_date": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(output_file, "w") as f:
        json.dump(info, f, indent=2)

    print(f"✓ Saved metadata: {output_file}")


def print_summary(
    output_dir: Path, molecules: List[Dict], activities: Optional[List[Dict]] = None
) -> None:
    """Print download summary"""
    print()
    print("=" * 70)
    print("DOWNLOAD SUMMARY")
    print("=" * 70)
    print()
    print(f"Output directory: {output_dir}")
    print(f"Molecules downloaded: {len(molecules):,}")
    if activities:
        print(f"Activities downloaded: {len(activities):,}")
    print()
    print("Files created:")
    print("  • molecules.smi - SMILES file")
    print("  • molecules.json - Full molecule data (JSON)")
    print("  • molecules.tsv - Molecule summary (TSV)")
    if activities:
        print("  • activities.tsv - Bioactivity data")
        print("  • targets.tsv - Target information")
    print("  • download_info.json - Download metadata")
    print()


# =============================================================================
# MAIN
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Download Molecule Data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python download_mol.py --source chembl --max-molecules 10000
    python download_mol.py --source chembl --include-activities --target-organism "Homo sapiens"
    python download_mol.py --smiles-file molecules.smi
    python download_mol.py --chembl-ids CHEMBL25,CHEMBL192,CHEMBL941

Sources:
    chembl  - ChEMBL database (default)
    pubchem - PubChem database
        """,
    )

    parser.add_argument(
        "--source",
        type=str,
        default="chembl",
        choices=["chembl", "pubchem"],
        help="Data source (default: chembl)",
    )

    parser.add_argument(
        "--max-molecules",
        type=int,
        default=10000,
        help="Maximum number of molecules to download (default: 10000)",
    )

    parser.add_argument(
        "--include-activities",
        action="store_true",
        help="Include bioactivity data from ChEMBL",
    )

    parser.add_argument(
        "--include-mechanisms",
        action="store_true",
        help="Also fetch mechanism-of-action data from ChEMBL (requires --include-activities)",
    )

    parser.add_argument(
        "--max-activities",
        type=int,
        default=50000,
        help="Maximum number of activities to download (default: 50000)",
    )

    parser.add_argument(
        "--target-organism",
        type=str,
        default="Homo sapiens",
        help="Filter activities by target organism (default: 'Homo sapiens')",
    )

    parser.add_argument(
        "--smiles-file",
        type=str,
        help="Load molecules from SMILES file instead of downloading",
    )

    parser.add_argument(
        "--chembl-ids",
        type=str,
        help="Comma-separated list of ChEMBL IDs to download",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/molecules",
        help="Output directory (default: data/molecules)",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("MOLECULE DATA DOWNLOAD")
    print("=" * 70)
    print()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {output_dir}")
    print()

    molecules: List[Dict] = []
    activities: List[Dict] = []
    targets: Dict[str, Dict] = {}

    # Load from file if specified
    if args.smiles_file:
        molecules = load_smiles_file(Path(args.smiles_file))

    # Fetch specific ChEMBL IDs
    elif args.chembl_ids:
        chembl_ids = [cid.strip() for cid in args.chembl_ids.split(",")]
        molecules = fetch_chembl_by_ids(chembl_ids)

    # Fetch from database
    elif args.source == "chembl":
        if args.include_activities:
            # Fetch activities first, then get molecules
            activities = fetch_chembl_activities(
                target_organism=args.target_organism,
                max_activities=args.max_activities,
            )

            # Get unique target IDs
            target_ids = list(
                set(
                    a["target_chembl_id"]
                    for a in activities
                    if a.get("target_chembl_id")
                )
            )

            # Fetch molecule SMILES
            smiles_dict = fetch_molecules_for_activities(activities)

            # Create molecule list
            for mol_id, smiles in smiles_dict.items():
                molecules.append({"chembl_id": mol_id, "smiles": smiles})

            # Fetch target info
            targets = fetch_target_info(target_ids)

        else:
            # Just fetch molecules
            molecules = fetch_chembl_molecules(max_molecules=args.max_molecules)

    if not molecules:
        print("❌ No molecules downloaded")
        return

    # Save data
    save_molecules(
        molecules,
        output_dir,
        activities if activities else None,
        targets if targets else None,
    )

    molecule_ids = list({a["molecule_chembl_id"] for a in activities if a.get("molecule_chembl_id")})
    if args.include_mechanisms and molecule_ids:
        mechanisms = fetch_mechanism_data(molecule_ids)
        mechanisms_file = output_dir / "mechanisms.json"
        with open(mechanisms_file, "w") as f:
            json.dump(mechanisms, f, indent=2)
        print(f"\n✓ Saved mechanisms to: {mechanisms_file}")

    # Save metadata
    save_metadata(
        output_dir=output_dir,
        source=args.source,
        molecule_count=len(molecules),
        activity_count=len(activities),
        target_organism=args.target_organism if args.include_activities else None,
    )

    # Print summary
    print_summary(output_dir, molecules, activities if activities else None)

    print("Done!")
    print()
    print("This molecule data can be used with:")
    print("  • download_p2m.py - Protein-molecule interactions")


if __name__ == "__main__":
    main()
