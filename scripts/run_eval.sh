#!/bin/bash

# Configuration
SCRIPT_PATH="./eval_parallel.py"  # Path to your Python script
LOG_DIR="logs"
MAX_WORKERS=4
DOMAINS=("legal" "financial" "academic")

# Optional API settings (uncomment and configure if needed)
# API_KEY="your_api_key_here"
# BASE_URL="your_base_url_here"

# Evaluation configurations
# Format: "MODEL_NAME;OTHERS_SETTING;PROMPT_SETTING"
# Use empty string for OTHERS_SETTING if not needed
EVAL_CONFIGS=(
    "deepseek/deepseek-r1-distill-llama-70b;;no_teach"
    "deepseek/deepseek-r1-distill-llama-70b;rag;all"
    # Add more configurations as needed
)

# Main execution
mkdir -p "${LOG_DIR}"

echo "Starting evaluation with ${#EVAL_CONFIGS[@]} configurations..."
echo "Logs will be saved to: ${LOG_DIR}"
echo "=================================="

for i in "${!EVAL_CONFIGS[@]}"; do
    config="${EVAL_CONFIGS[$i]}"
    IFS=';' read -r model others prompt <<< "$config"
    
    echo "[$((i+1))/${#EVAL_CONFIGS[@]}] Running: $model | $others | $prompt"
    
    # Create log filename
    timestamp=$(date +"%Y%m%d_%H%M%S")
    safe_model=${model//\//_}
    others_log=${others:-"none"}
    others_log=${others_log// /_}
    log_file="${LOG_DIR}/run_${safe_model}_${prompt}_${others_log}_${timestamp}.log"
    
    # Run evaluation
    python "${SCRIPT_PATH}" \
        --model-name "${model}" \
        --prompt_setting "${prompt}" \
        --max_workers "${MAX_WORKERS}" \
        --others "${others}" \
        --domains "${DOMAINS[@]}" \
        > "${log_file}" 2>&1
        # --api_key "${API_KEY}" \
        # --base_url "${BASE_URL}" \
    
    if [ $? -eq 0 ]; then
        echo "✓ SUCCESS - Log: ${log_file}"
    else
        echo "✗ FAILED - Check: ${log_file}"
    fi
    echo ""
done

echo "=================================="
echo "All evaluations completed!"
echo "Check logs in: ${LOG_DIR}"