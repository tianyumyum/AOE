#!/bin/bash

# =============================================================================
# LLM Generation Runner
# Runs parallel generation tasks with different model configurations
# =============================================================================

# Configuration
SCRIPT_PATH=""  # Path to your generate_parallel.py script
LOG_DIR_BASE="logs_$(date +%Y%m%d)"
MAX_WORKERS=4
DOMAINS=("legal" "financial" "academic")

# Model settings (set your values here)
MODEL_NAME="your-model-name"
BASE_URL=""
API_KEY=""

# Evaluation configurations: "MODEL_NAME;OTHERS_SETTING;PROMPT_SETTING"
# Use semicolon as delimiter, empty OTHERS as "model;;prompt"
EVAL_CONFIGURATIONS=(
    "${MODEL_NAME};;baseline"
    "${MODEL_NAME};rag;all"
    # Add more configurations as needed
)

# =============================================================================

# Validation
if [ -z "${SCRIPT_PATH}" ] || [ ! -f "${SCRIPT_PATH}" ]; then
    echo "ERROR: Set SCRIPT_PATH to your generate_parallel.py script"
    exit 1
fi

mkdir -p "${LOG_DIR_BASE}"
TOTAL_CONFIGS=${#EVAL_CONFIGURATIONS[@]}

echo "Starting LLM generation with ${TOTAL_CONFIGS} configurations"
echo "Log directory: ${LOG_DIR_BASE}"
echo ""

# Process configurations
for i in "${!EVAL_CONFIGURATIONS[@]}"; do
    config="${EVAL_CONFIGURATIONS[$i]}"
    IFS=';' read -r model_name others_setting prompt_setting <<< "$config"
    
    echo "[$((i+1))/${TOTAL_CONFIGS}] Running: ${model_name} | ${others_setting:-none} | ${prompt_setting}"
    
    # Generate log filename
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    SAFE_MODEL=${model_name//\//_}
    OTHERS_SAFE=${others_setting// /_}
    OTHERS_SAFE=${OTHERS_SAFE:-none}
    
    LOG_FILE="${LOG_DIR_BASE}/run_${SAFE_MODEL}_${prompt_setting}_${OTHERS_SAFE}_${TIMESTAMP}.log"
    
    # Build command
    CMD_ARGS=(
        "${SCRIPT_PATH}"
        --model-name "${model_name}"
        --prompt_setting "${prompt_setting}"
        --max_workers "${MAX_WORKERS}"
        --others "${others_setting}"
        --domains "${DOMAINS[@]}"
    )
    
    # Add API settings if provided
    [ -n "${API_KEY}" ] && CMD_ARGS+=(--api_key "${API_KEY}")
    [ -n "${BASE_URL}" ] && CMD_ARGS+=(--base-url "${BASE_URL}")
    
    # Execute
    if python "${CMD_ARGS[@]}" > "${LOG_FILE}" 2>&1; then
        echo "✓ SUCCESS"
    else
        echo "✗ FAILED (check ${LOG_FILE})"
    fi
    echo ""
done

echo "All configurations processed. Logs in: ${LOG_DIR_BASE}"