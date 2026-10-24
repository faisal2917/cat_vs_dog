# Cat vs Dog classifier

Trains a cat/dog image classifier on a laptop CPU with **transfer learning**: it starts from
MobileNetV3-Large, which is already trained on ImageNet's 1.2 million photos, and teaches it
the two new classes. That takes about 20 minutes on a ThinkPad X13 Yoga instead of days, and
should reach about 98–99% accuracy.

| File | What it does |
|---|---|
| `prepare_data.py` | Downloads the dataset, drops broken images, shrinks them, and splits them 80/10/10 into train/val/test |
| `train.py` | Fine-tunes the model and saves the best one to `models/cat_dog_mobilenetv3.pt` |
| `train_scratch.py` | Alternative: trains a small CNN from random weights (hours, less accurate, see [Training from scratch](#training-from-scratch)) |
| `evaluate.py` | Scores a trained model on the test images and lists its mistakes |
| `predict.py` | Labels your own photos as cat or dog |
| `app.py`, `static/` | Web page + API: upload a photo, get cat or dog (see [Docker](#docker)) |
| `Dockerfile`, `docker-compose.yml` | Run the web app, or any of the scripts, in Docker |

## Your machine

ThinkPad X13 Yoga Gen 1: Intel i5-10210U (4 cores, 15 W), 16 GB RAM, Intel UHD Graphics.
The integrated GPU can't be used by PyTorch (it has no CUDA, and Intel's PyTorch GPU support
only covers Arc and Core Ultra graphics), so everything runs on the CPU. That is why `train.py`
uses a small pretrained model; training from scratch (see below) takes hours instead.

## Steps

Run these in PowerShell from this folder (`D:\faisal\data_science\cat_vs_dog`).

**1. Set up Python packages** (already done: `.venv` exists and has these installed)

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

**2. Activate the environment** (do this in every new terminal)

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell says running scripts is disabled, skip activation and write
`.\.venv\Scripts\python.exe` wherever the steps below say `python`.

**3. Download and prepare the data**

```powershell
python prepare_data.py
```

This downloads 825 MB to `data\` and creates `data\processed\{train,val,test}\{cat,dog}\`.

**4. Quick test run** (about 2 minutes; checks everything works before the long run)

```powershell
python train.py --subset 500
```

**5. Full training**

```powershell
python train.py
```

Two epochs train only the new last layer, then one epoch also fine-tunes the top 4 backbone
blocks. The best model is saved after every epoch, so Ctrl+C still leaves you a usable model.
At the end it prints the accuracy on 2,500 test images it has never seen.

**6. Test it**

Both scripts use the most recently trained model in `models\` unless you pass
`--model models\<file>.pt`.

Score it on the ~2,500 test images that were held back from training:

```powershell
python evaluate.py
```

It prints the accuracy, a table of how many cats and dogs it got right or wrong, and the
mistakes it was most sure about, with file paths so you can open them. Some of those turn out
to be labelling errors in the dataset (for example a photo of a cat filed under dog), so the
true accuracy is slightly higher than the number shown.

Try it on your own photos (phone pictures, images from the web):

```powershell
python predict.py C:\path\to\photo.jpg
python predict.py C:\path\to\folder_of_photos
```

Output looks like ` cat   99.7%  photo.jpg`. The model only knows two classes, so a
photo of anything else (a car, a person) still gets labelled cat or dog.

To test on a whole folder of your own labelled photos, put them in `my_test\cat\` and
`my_test\dog\`, then run `python evaluate.py --data my_test`.

## Time estimates on this laptop

The speeds were measured on this laptop **on battery, in Balanced power mode**, at the CPU's
sustained (throttled) speed. Plugged in with Best performance it should be somewhat faster.
Train set ≈ 20,000 images.

| Step | Time |
|---|---|
| Install packages | done (about 5 min) |
| Download dataset, 825 MB | ~6 min at your connection's measured ~2.4 MB/s |
| Extract + clean + resize 25,000 images | ~3–5 min |
| Quick test, `--subset 500` | ~2 min |
| **Full training, default (224 px)**: 2 × ~6.5 min + ~8 min + test | **~20–30 min** |
| Faster training, `--img-size 160` | ~12–18 min |
| Predicting one photo | well under a second |

Measured training speed: 224 px ≈ 55 img/s (last layer only) and ≈ 44 img/s (fine-tuning);
160 px ≈ 84 and ≈ 80 img/s. With heavy apps open (browser, Teams, Docker) expect it to take longer.

Useful `train.py` options:

| Option | Default | Effect |
|---|---|---|
| `--img-size 160` | 224 | ~1.7× faster, slightly less accurate |
| `--head-epochs N` | 2 | epochs that train only the new last layer (~6.5 min each) |
| `--finetune-epochs N` | 1 | epochs that also fine-tune the top blocks (~8 min each); 0 skips this phase |
| `--subset N` | all | use only N training images per class |
| `--batch-size N` | 32 | lower it if you run short of memory |
| `--workers N` | 3 | processes that load images in parallel |

### Tips for training on a laptop

- Plug in the charger and set Windows power mode to **Best performance**
  (Settings → System → Power & battery).
- Stop the laptop from sleeping during training (Screen and sleep → Never, while plugged in).
- Close heavy apps (browser, Teams). Training uses all 4 cores.
- The fan will run at full speed. That is normal; keep the vents uncovered.

## Training from scratch

`train_scratch.py` trains a small CNN (9 conv layers, 1.2M parameters, 128×128 input) that
starts with random weights and learns only from these 25,000 images. Every layer has to learn,
so it needs many more epochs than `train.py`. It is slower and ends less accurate, but it shows
what a CNN can learn on its own.

Use the same data as above (step 3), then:

```powershell
python train_scratch.py --subset 500 --epochs 2   # quick test, ~2 min
python train_scratch.py                           # full run, 30 epochs
python predict.py photo.jpg --model models\cat_dog_scratch.pt
```

A checkpoint is written after every epoch. If training stops (Ctrl+C, sleep, crash, flat
battery), run the same command again with `--resume` and it continues from the last finished
epoch. Each epoch line also prints roughly how many minutes are left.

Measured speed: ~60 img/s (on battery, with augmentation and image loading), so
**~6 min per epoch**.

| Run | Time on this laptop | Test accuracy |
|---|---|---|
| `--epochs 15` | ~1.5 h | not measured |
| **default, 30 epochs** | **~2.5 h (measured)** | **97.2% (measured, 2,499 test images)** |
| `--epochs 50` | ~4–5 h | not measured |
| `--img-size 160`, 30 epochs | ~4–5 h | not measured |
| For comparison: `train.py` (pretrained) | ~20–30 min | ~98–99% expected |

Training the bigger MobileNetV3 from scratch was also measured: ~17 img/s at 224 px, about
20 min per epoch, or ~10 hours for 30 epochs. That's why the small CNN is used instead.

For a run of several hours: plug in the charger, set sleep to **Never** while plugged in, and
leave the lid open (or set "When I close the lid" to *Do nothing* in Control Panel → Power
Options). If it does sleep, just `--resume`.

## Docker

Docker Desktop must be running first (whale icon in the taskbar says "Engine running").

**Web app**: start it, then open http://localhost:8000 and choose a photo:

```powershell
docker compose up --build -d
```

The first build downloads Python and PyTorch (CPU version), takes a few minutes, and makes an
image of about 1 GB. Later starts take seconds. The app uses the newest model in `models\`; it
reads it at start-up, so after training a new one run `docker compose restart web`.

```powershell
docker compose logs -f web      # see what it's doing (Ctrl+C stops watching, not the app)
docker compose down             # stop it
```

The API can be called from any program, for example:

```powershell
curl.exe -F "file=@C:\path\to\photo.jpg" http://localhost:8000/predict
# {"label":"cat","confidence":0.9731,"probabilities":{"cat":0.9731,"dog":0.0269}}
```

The app only listens on this computer. To open it from your phone on the same Wi-Fi, change
`"127.0.0.1:8000:8000"` to `"8000:8000"` in `docker-compose.yml` and browse to
`http://<laptop IP>:8000`.

**Training and testing in Docker**: the `trainer` service runs any script with `data\` and
`models\` shared with this folder:

```powershell
docker compose run --rm trainer python prepare_data.py
docker compose run --rm trainer                              # train_scratch.py
docker compose run --rm trainer python train_scratch.py --resume
docker compose run --rm trainer python train.py
docker compose run --rm trainer python evaluate.py
```

On this laptop, training in Docker is slower than with `.venv`, because the container reads
25,000 images from the Windows drive through Docker's file sharing. Docker training is mainly
useful for moving the project to another machine or a Linux server.

To run the web app without Docker: `python -m uvicorn app:app --port 8000`.

**If Docker Desktop shows an error with "The file cannot be accessed by the system"** (it
can't delete its own socket files in `%LOCALAPPDATA%\Docker\run` or
`%LOCALAPPDATA%\docker-secrets-engine`): restart Windows, then start Docker Desktop again. Don't
choose "Reset to factory defaults"; that deletes all your images and containers.

## Data sources

| Dataset | Size | Notes |
|---|---|---|
| **[Microsoft Kaggle Cats and Dogs](https://www.microsoft.com/en-us/download/details.aspx?id=54765)** (used by default) | 25,000 images, 825 MB | 12,500 per class, no account needed, CDLA-Permissive-2.0 licence. Contains a few broken files, which `prepare_data.py` skips |
| [Kaggle "Dogs vs. Cats"](https://www.kaggle.com/c/dogs-vs-cats/data) | the same 25,000 images | Needs a Kaggle account. Unzip its inner `train.zip`, then run `python prepare_data.py --src path\to\train` |
| [Hugging Face `microsoft/cats_vs_dogs`](https://huggingface.co/datasets/microsoft/cats_vs_dogs) | 23,410 images | The Microsoft set, already cleaned, for use with the `datasets` library |
| [Oxford-IIIT Pet](https://www.robots.ox.ac.uk/~vgg/data/pets/) | 7,349 images, 37 breeds | Higher-quality photos, good as an extra test set. Also available as `torchvision.datasets.OxfordIIITPet(target_types="binary-category")` |
| Your own photos | any | Put them in `cat\` and `dog\` folders, then run `python prepare_data.py --src that_folder` |

`prepare_data.py --zip file.zip` uses a zip you already downloaded instead of downloading
again.

## Faster alternative

`train.py` uses an NVIDIA GPU automatically when one is available. If you want to experiment
a lot, the free GPUs in [Google Colab](https://colab.research.google.com) or
[Kaggle Notebooks](https://www.kaggle.com/code) should train in roughly 5 minutes (not measured
here). Upload the scripts, run `prepare_data.py`, then `train.py --workers 2`.

## Troubleshooting

- **`No cat/dog images found`**: the `--src` folder needs images in `cat`/`dog` (or `Cat`/`Dog`,
  `cats`/`dogs`) folders, or filenames that start with `cat.`/`dog.`.
- **Training is much slower than the table**: check the charger and power mode, or try
  `--workers 2` or `--img-size 160`.
- **First run pauses at "Downloading mobilenet_v3_large"**: it fetches the 21 MB pretrained
  weights once and caches them in `%USERPROFILE%\.cache\torch`.
#   c a t _ v s _ d o g  
 