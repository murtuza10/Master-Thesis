#!/bin/bash
#SBATCH --partition=mlgpu_short
#SBATCH --time=8:00:00
#SBATCH --gpus=16
#SBATCH --output=/home/s27mhusa_hpc/Master-Thesis/FinalOutput-3rdSeptember-FourShotPrompting_Final/gpu_job_TestFiles_%j_3rdSeptember.out

MODEL_NAME=$1

INPUT_DIR="/home/s27mhusa_hpc/Master-Thesis/Text_Files_Test_Data"
OUTPUT_DIR="/home/s27mhusa_hpc/Master-Thesis/Results/FinalResults_TestFiles_3rdSeptember_Prompting/LLM_annotated_${MODEL_NAME}_4shot"
OUTPUT_DIR_JSON="/home/s27mhusa_hpc/Master-Thesis/Results/FinalResults_TestFiles_3rdSeptember_Prompting_json/LLM_annotated_${MODEL_NAME}_4shot"
MODEL_PATH="/lustre/scratch/data/s27mhusa_hpc-murtuza_master_thesis/${MODEL_NAME}"
START=4

module load CUDA/12.6.0
module load sysstat


source ~/.bashrc
conda init
conda deactivate 
conda activate Llama

# Record start time
start_time=$(date +%s)

# === Begin Resource Monitoring ===
LOG_DIR="/home/s27mhusa_hpc/Master-Thesis/FinalOutput-3rdSeptember-FourShotPrompting_Final/job_monitor_logs_TestFiles${MODEL_NAME}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
INTERVAL=1

echo "Starting resource monitoring every $INTERVAL seconds..."
echo "Logs will be stored in: $LOG_DIR"


export LOG_DIR="$LOG_DIR"

# GPU Monitoring
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,temperature.gpu,power.draw \
  --format=csv -l $INTERVAL > "$LOG_DIR/gpu_usage.log" &
GPU_PID=$!

# CPU & Memory Monitoring
(
  while true; do
    echo "$(date)" >> "$LOG_DIR/cpu_mem.log"
    top -b -n1 | head -n 20 >> "$LOG_DIR/cpu_mem.log"
    echo "===============================" >> "$LOG_DIR/cpu_mem.log"
    sleep $INTERVAL
  done
) &
CPU_PID=$!

# Disk I/O Monitoring
iostat -xz $INTERVAL > "$LOG_DIR/disk_io.log" &
IO_PID=$!

# Save monitor PIDs
echo $GPU_PID > "$LOG_DIR/pids.txt"
echo $CPU_PID >> "$LOG_DIR/pids.txt"
echo $IO_PID >> "$LOG_DIR/pids.txt"

# === Run Main Script ===
python /home/s27mhusa_hpc/Master-Thesis/FewShotPrompting/NER_Prompting_generic_chat_Fourshot.py \
  --input_dir "$INPUT_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --output_dir_json "$OUTPUT_DIR_JSON" \
  --model_name "$MODEL_NAME" \
  --model_path "$MODEL_PATH" \
  --start $START \
  --max_length 4000

# === Stop Resource Monitoring ===
echo "Stopping resource monitoring..."
kill $GPU_PID
kill $CPU_PID
kill $IO_PID

# Record end time
end_time=$(date +%s)
elapsed=$(( end_time - start_time ))

echo "Total GPU execution time: $elapsed seconds"
echo "$elapsed" > "$LOG_DIR/elapsed_time_seconds.txt"

# === Save execution time into Excel sheet ===
EXCEL_PATH="/home/s27mhusa_hpc/Master-Thesis/Evaluation_Results/Final_TestFiles_3rdSeptember_FewShotTest_Broad/ner_evaluation_results_${MODEL_NAME}_${START}_shot.xlsx"

python <<EOF
import openpyxl
import os

excel_path = "$EXCEL_PATH"
execution_time = $elapsed  # seconds
examples = "$EXAMPLES"

# Ensure Excel file exists
if not os.path.exists(excel_path):
    # Create new workbook if missing
    from openpyxl import Workbook
    wb = Workbook()
    ws = wb.active
    ws.append(["Model", "Examples", "OtherMetrics...", "Time (s)"])
    wb.save(excel_path)

# Load workbook
wb = openpyxl.load_workbook(excel_path)
ws = wb.active

# Try to find row matching model+examples
row_to_update = None
for row in ws.iter_rows(min_row=2, values_only=False):  # skip header
    model_val = row[1].value
    if str(model_val) == "$MODEL_NAME":
        row_to_update = row
        break

if row_to_update:
    # Update the "Time" column (last column)
  for cell in row_to_update:
    if cell.value is None:
      cell.value = execution_time
      break
  else:
    col_index = ws.max_column + 1
    ws.cell(row=row_to_update[0].row, column=col_index, value=execution_time)

wb.save(excel_path)
print(f"✔ Execution time saved to Excel: {excel_path}")
EOF



# === SLURM Summary Metrics ===
echo "Fetching SLURM job summary..."
sacct -j $SLURM_JOB_ID --format=JobID,Elapsed,MaxRSS,CPUTime,ExitCode > "$LOG_DIR/slurm_summary.log"
cat "$LOG_DIR/slurm_summary.log"

echo "All logs are saved in: $LOG_DIR"
