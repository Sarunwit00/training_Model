# Training preset for RTX 3050 4GB Laptop
# Run: .\train_rtx3050_4gb.ps1

.\venv\Scripts\Activate.ps1

python scripts/train_wav2vec2.py `
    --base_model airesearch/wav2vec2-large-xlsr-53-th `
    --target text_dialect `
    --output_dir models/wav2vec2-south-th-fast `
    --epochs 5 `
    --batch_size 1 `
    --gradient_accumulation_steps 8 `
    --lr 1e-4 `
    --warmup_steps 100 `
    --eval_steps 200 `
    --save_steps 200 `
    --weight_decay 0.005 `
    --max_audio_seconds 3.5 `
    --max_train_samples 1000 `
    --num_workers 0
