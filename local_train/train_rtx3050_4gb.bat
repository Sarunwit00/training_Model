@echo off
REM ============================================================
REM   Training preset for RTX 3050 4GB Laptop
REM   - Use small effective batch (1 x 8 grad accum = 8)
REM   - Filter long clips (>3.5s) to save VRAM
REM   - Subsample train to 1000 clips for fast iteration
REM   - 5 epochs only (focus on results not perfection)
REM   - Expected time: 1-2 hours on RTX 3050 mobile
REM ============================================================

call venv\Scripts\activate.bat

python scripts/train_wav2vec2.py ^
    --base_model airesearch/wav2vec2-large-xlsr-53-th ^
    --target text_dialect ^
    --output_dir models/wav2vec2-south-th-fast ^
    --epochs 5 ^
    --batch_size 1 ^
    --gradient_accumulation_steps 8 ^
    --lr 1e-4 ^
    --warmup_steps 100 ^
    --eval_steps 200 ^
    --save_steps 200 ^
    --weight_decay 0.005 ^
    --max_audio_seconds 3.5 ^
    --max_train_samples 1000 ^
    --num_workers 0

pause
