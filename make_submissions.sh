#!/usr/bin/env bash
# Run from starting_kit/:  bash make_submissions.sh
# For a single model:      bash make_submissions.sh rasch
set -euo pipefail
cd "$(dirname "$0")"

TOOLS="tools/run_smoke_test.py"
CHECK="tools/check_submission_zip.py"

submit() {
    local label=$1
    local dir=$2
    shift 2
    local files=("$@")

    echo "━━━ $label ━━━"

    # Smoke test against the unpacked directory
    python "$TOOLS" "$dir"

    # Zip only the inference files (no train.py, _data.py, val_split.pkl, etc.)
    local zipfile="${label}.zip"
    rm -f "$zipfile"
    local fullpaths=()
    for f in "${files[@]}"; do
        fullpaths+=("$dir/$f")
    done
    zip -j "$zipfile" "${fullpaths[@]}"

    # Validate the zip
    python "$CHECK" "$zipfile"

    echo "Created $zipfile"
    echo ""
}

TARGET="${1:-all}"

run_rasch()                  { submit rasch                   rasch_submission                  model.py rasch.pt; }
run_twopl()                  { submit twopl                   twopl_submission                  model.py twopl.pt; }
run_threepl()                { submit threepl                 threepl_submission                model.py threepl.pt; }
run_amortized_irt()          { submit amortized_irt           amortized_irt_submission          model.py amortized_irt.pt models.txt; }
run_amortized_irt_rasch()    { submit amortized_irt_rasch     amortized_irt_rasch_submission    model.py amortized_irt_rasch.pt models.txt; }
run_amortized_tfidf()        { submit amortized_tfidf         amortized_tfidf_submission        model.py amortized_irt_tfidf.pt tfidf_arrays.npz; }
run_amortized_rasch_tfidf()  { submit amortized_rasch_tfidf   amortized_rasch_tfidf_submission  model.py amortized_rasch_tfidf.pt tfidf_arrays.npz; }
run_amortized_3pl_sentence() { submit amortized_3pl_sentence  amortized_3pl_sentence_submission model.py amortized_3pl_sentence.pt models.txt; }
run_amortized_3pl_tfidf()    { submit amortized_3pl_tfidf     amortized_3pl_tfidf_submission    model.py amortized_3pl_tfidf.pt tfidf_arrays.npz; }
run_multifacet()             { submit multifacet              multifacet_submission             model.py multifacet_2pl.pt; }
run_ncf()                    { submit ncf                     ncf_submission                    model.py ncf_head.pt ncf_meta_slim.pkl models.txt; }

case "$TARGET" in
    all)
        run_rasch
        run_twopl
        run_threepl
        run_amortized_irt
        run_amortized_irt_rasch
        run_amortized_tfidf
        run_amortized_rasch_tfidf
        run_amortized_3pl_sentence
        run_amortized_3pl_tfidf
        run_multifacet
        run_ncf
        ;;
    rasch)                   run_rasch ;;
    twopl)                   run_twopl ;;
    threepl)                 run_threepl ;;
    amortized_irt)           run_amortized_irt ;;
    amortized_irt_rasch)     run_amortized_irt_rasch ;;
    amortized_tfidf)         run_amortized_tfidf ;;
    amortized_rasch_tfidf)   run_amortized_rasch_tfidf ;;
    amortized_3pl_sentence)  run_amortized_3pl_sentence ;;
    amortized_3pl_tfidf)     run_amortized_3pl_tfidf ;;
    multifacet)              run_multifacet ;;
    ncf)                     run_ncf ;;
    *)
        echo "Unknown target: $TARGET"
        echo "Usage: bash make_submissions.sh [all|rasch|twopl|threepl|amortized_irt|amortized_irt_rasch|amortized_tfidf|amortized_rasch_tfidf|amortized_3pl_sentence|amortized_3pl_tfidf|multifacet|ncf]"
        exit 1
        ;;
esac
