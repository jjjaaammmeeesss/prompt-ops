"""
Convert all 82 parent-child dialogue test case files into a single JSON array
for the prompt-ops dataset pipeline.

Source directories:
  1. batch1 (22 files)
  2. batch2 (60 files)

Output: data/test_cases_questions.json
"""

import json
import os
import sys

# === Config ===
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_PATH = os.path.join(DATA_DIR, 'test_cases_questions.json')

SOURCE_DIRS = [
    r'C:\Users\h\Downloads\7.6 test_cases\test_cases',
    r'C:\Users\h\Downloads\第二批-7.6（60）\第二批-7.6（60）',
]


def find_txt_files(root_dir):
    """Recursively find all .txt files under root_dir."""
    txt_files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname.endswith('.txt'):
                txt_files.append(os.path.join(dirpath, fname))
    return txt_files


def process_file(filepath, batch_label):
    """
    Read a test-case file and extract the dialogue.

    Returns: (question_dict, None) on success, or (None, warning_str) on skip.
    """
    filename = os.path.basename(filepath)
    # Category is the parent directory name (e.g. "A优秀对话")
    category = os.path.basename(os.path.dirname(filepath))

    with open(filepath, 'r', encoding='utf-8-sig') as f:
        lines = f.readlines()

    # Strip lines starting with '#' (metadata headers)
    dialogue_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('#'):
            continue
        dialogue_lines.append(line.rstrip('\n'))

    # Join dialogue lines with \n
    dialogue = '\n'.join(dialogue_lines)

    # Trim leading/trailing whitespace
    dialogue = dialogue.strip()

    if not dialogue:
        return None, f"{batch_label}/{category}/{filename}: no dialogue content after stripping metadata"

    source = f"{batch_label}_{filename}"
    return {
        'question': dialogue,
        'category': category,
        'source': source,
    }, None


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    all_questions = []
    warnings = []
    total_found = 0
    total_converted = 0
    total_skipped = 0

    batch_labels = ['batch1', 'batch2']

    for batch_idx, src_dir in enumerate(SOURCE_DIRS):
        batch_label = batch_labels[batch_idx]
        if not os.path.isdir(src_dir):
            print(f'WARNING: source directory not found: {src_dir}', file=sys.stderr)
            continue

        txt_files = find_txt_files(src_dir)
        total_found += len(txt_files)
        print(f'[{batch_label}] Found {len(txt_files)} .txt files in {src_dir}')

        for filepath in sorted(txt_files):
            result, warning = process_file(filepath, batch_label)
            if result:
                all_questions.append(result)
                total_converted += 1
            if warning:
                warnings.append(warning)
                total_skipped += 1

    # Print warnings to stderr
    for w in warnings:
        print(f'WARNING: {w}', file=sys.stderr)

    # Verification
    null_empty = [q for q in all_questions if not q.get('question')]
    if null_empty:
        print(
            f'ERROR: {len(null_empty)} entries have null/empty question field',
            file=sys.stderr,
        )

    # Write output
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(all_questions, f, ensure_ascii=False, indent=2)

    # Summary
    print(f'\n{"=" * 50}')
    print(f'SUMMARY')
    print(f'{"=" * 50}')
    print(f'Total .txt files found: {total_found}')
    print(f'Total converted:        {total_converted}')
    print(f'Total skipped:          {total_skipped}')
    print(f'Final JSON entries:     {len(all_questions)}')
    print(f'Verification:           {"PASS" if len(all_questions) == 82 and not null_empty else "FAIL"}')
    print(f'Output:                 {OUTPUT_PATH}')
    print(f'Output size:            {os.path.getsize(OUTPUT_PATH)} bytes')

    if len(all_questions) != 82:
        print(f'\nWARNING: Expected 82 entries, got {len(all_questions)}', file=sys.stderr)
        sys.exit(1)

    if null_empty:
        sys.exit(1)


if __name__ == '__main__':
    main()
