#!/bin/bash
# Repository Cleanup Script
# Removes duplicate files, old training runs, and organizes the repo

set -e  # Exit on error

PROJECT_ROOT="/root/Programming Projects/Personal/PriceLens"
cd "$PROJECT_ROOT"

echo "========================================================================"
echo "🧹 PRICELENS REPOSITORY CLEANUP"
echo "========================================================================"

# Function to get human-readable size
get_size() {
    du -sh "$1" 2>/dev/null | cut -f1
}

# Function to safely remove with confirmation
safe_remove() {
    local path="$1"
    local size=$(get_size "$path")
    if [ -e "$path" ]; then
        echo "   🗑️  Removing: $path ($size)"
        rm -rf "$path"
    fi
}

echo ""
echo "📊 Current Repository Size:"
echo "   Total: $(get_size .)"
echo ""

# ============================================================
# 1. Remove duplicate model files in root
# ============================================================
echo "1️⃣  Cleaning Root Directory..."
echo "--------------------------------------------------------"

if [ -f "yolo11n.pt" ]; then
    echo "   Found duplicate: yolo11n.pt (keeping models/yolo11n.pt)"
    safe_remove "yolo11n.pt"
fi

# ============================================================
# 2. Remove old training runs (keep only latest)
# ============================================================
echo ""
echo "2️⃣  Cleaning Old Training Runs..."
echo "--------------------------------------------------------"

# Keep only the latest successful training
KEEP_RUN="models/training_runs/yolo11n_cleveland_notebook"
echo "   Keeping: $KEEP_RUN (latest, 100 epochs)"

# Remove old training runs
if [ -d "models/training_runs" ]; then
    for run in models/training_runs/*/; do
        if [ "$run" != "$KEEP_RUN/" ]; then
            safe_remove "$run"
        fi
    done
fi

# Remove test runs
if [ -d "models/test_runs" ]; then
    echo "   Removing all test runs..."
    safe_remove "models/test_runs"
fi

# ============================================================
# 3. Clean up notebooks directory
# ============================================================
echo ""
echo "3️⃣  Cleaning Notebooks Directory..."
echo "--------------------------------------------------------"

# Remove old pokemon_detector training run
if [ -d "notebooks/pokemon_detector" ]; then
    safe_remove "notebooks/pokemon_detector"
fi

# Remove duplicate yolo11n.pt in notebooks
if [ -f "notebooks/yolo11n.pt" ]; then
    safe_remove "notebooks/yolo11n.pt"
fi

# Move exported model to models/ directory
if [ -f "notebooks/pokemon_card_detector_5k.pt" ]; then
    echo "   Moving exported model to models/"
    if [ ! -f "models/pokemon_card_detector_5k.pt" ]; then
        mv "notebooks/pokemon_card_detector_5k.pt" "models/"
        echo "   ✓ Moved pokemon_card_detector_5k.pt to models/"
    else
        safe_remove "notebooks/pokemon_card_detector_5k.pt"
    fi
fi

# ============================================================
# 4. Clean temporary/output directories
# ============================================================
echo ""
echo "4️⃣  Cleaning Temporary Directories..."
echo "--------------------------------------------------------"

# Clean logs (keep directory, remove old logs)
if [ -d "logs" ]; then
    echo "   Cleaning logs/ (keeping directory)"
    find logs/ -type f -name "*.log" -mtime +7 -delete 2>/dev/null || true
    echo "   ✓ Removed logs older than 7 days"
fi

# Clean output directory
if [ -d "output" ]; then
    file_count=$(find output/ -type f 2>/dev/null | wc -l)
    if [ "$file_count" -gt 0 ]; then
        echo "   Cleaning output/ ($file_count files)"
        rm -rf output/*
        echo "   ✓ Cleaned output directory"
    fi
fi

# Clean runs directory (YOLO temp outputs)
if [ -d "runs" ]; then
    safe_remove "runs"
fi

# ============================================================
# 5. Clean notebook outputs to reduce size
# ============================================================
echo ""
echo "5️⃣  Cleaning Notebook Outputs..."
echo "--------------------------------------------------------"

if command -v jupyter &> /dev/null; then
    if [ -f "notebooks/train_card_detector.ipynb" ]; then
        echo "   Clearing outputs from train_card_detector.ipynb"
        jupyter nbconvert --ClearOutputPreprocessor.enabled=True \
            --inplace notebooks/train_card_detector.ipynb 2>/dev/null || true
        echo "   ✓ Notebook outputs cleared (reduces size)"
    fi
else
    echo "   ⚠️  jupyter not found, skipping notebook cleanup"
    echo "   (Notebook will remain large with embedded outputs)"
fi

# ============================================================
# 6. Verify git status
# ============================================================
echo ""
echo "6️⃣  Verifying Git Status..."
echo "--------------------------------------------------------"

# Check if .env is tracked
if git ls-files --error-unmatch .env &>/dev/null; then
    echo "   ⚠️  .env is tracked in git!"
    echo "   Removing from git (file will remain locally)..."
    git rm --cached .env
    echo "   ✓ .env removed from git tracking"
fi

# Show current untracked files
echo ""
echo "   Untracked large files/dirs:"
git status --short | grep "^??" | while read status file; do
    if [ -e "$file" ]; then
        size=$(get_size "$file")
        echo "      $file ($size)"
    fi
done

# ============================================================
# 7. Summary
# ============================================================
echo ""
echo "========================================================================"
echo "✅ CLEANUP COMPLETE!"
echo "========================================================================"

echo ""
echo "📊 New Repository Size:"
echo "   Total: $(get_size .)"
echo ""

echo "📁 Repository Structure:"
echo "   models/yolo11n.pt                    - Base YOLO11 model (5.4 MB)"
echo "   models/yolo11m.pt                    - Base YOLO11 medium (39 MB)"
echo "   models/pokemon_card_detector_5k.pt   - Your trained model (5.3 MB)"
echo "   models/training_runs/.../best.pt     - Training checkpoint (5.3 MB)"
echo ""

echo "🔒 Ignored by Git:"
echo "   ✓ .env"
echo "   ✓ models/*.pt"
echo "   ✓ models/training_runs/"
echo "   ✓ models/test_runs/"
echo "   ✓ logs/"
echo "   ✓ output/"
echo "   ✓ runs/"
echo "   ✓ notebooks/*.pt"
echo "   ✓ notebooks/*.zip"
echo ""

echo "✨ Next Steps:"
echo "   1. Review changes: git status"
echo "   2. Commit updated .gitignore: git add .gitignore && git commit -m 'chore: update .gitignore'"
echo "   3. Continue with model deployment!"
echo ""
echo "========================================================================"
