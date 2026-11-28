# Step 8 – The whole system in Docker

**Goal of this step:** start the live system with **one command**, the same way on any computer that
has Docker. No Python setup and no "it works on my machine".

Files:
- [`Dockerfile`](../Dockerfile): how the image is built.
- [`docker-compose.yml`](../docker-compose.yml): which services run, and how they connect.
- [`.dockerignore`](../.dockerignore): what is never sent into the image.

```bash
docker-compose up -d --build                     # build the image, start the API and the dashboard
docker-compose run --rm replay --dataset bgl     # start the live log feed (BGL test period)
docker-compose down                              # stop everything
```
- API: http://localhost:8000/docs
- Dashboard: http://localhost:8501

**Before the first start:**
- run scripts 01–06, which make the data, models and results;
- put `DATABASE_URL` in `.env` (see `.env.example`).

If these ports are taken: `API_PORT=18000 DASHBOARD_PORT=18501 docker-compose up -d`.

Older Docker installations use the separate `docker-compose` command, as shown here. With the newer
Compose plugin it is `docker compose` (with a space); everything else is the same.

---

## 1. Docker in plain words

- An **image** is a packed, read-only copy of everything a program needs: an operating system
  (here a small Debian Linux), Python, the libraries and our code.
- A **container** is a running copy of an image. It is isolated from the computer and from other
  containers, except for what we connect on purpose (ports, folders, a network).
- **Docker Compose** starts several containers together from one file (`docker-compose.yml`) and
  gives them a private network where they find each other by name (`http://api:8000`).

**Why it was promised in the project proposal:** the system then runs the same everywhere, with the same
Python and library versions as the evaluation, on a laptop, a server or in the cloud.

---

## 2. What runs

| Service | What it runs | Started by |
|---|---|---|
| `api` | The scoring API (`uvicorn src.api:app`), port 8000 | `docker-compose up` |
| `dashboard` | The Streamlit dashboard, port 8501; starts once the API is healthy | `docker-compose up` |
| `replay` | The live log feed (`python -m src.replay ...`); runs once, then stops | `docker-compose run --rm replay ...` |
| `db` | *Optional* local PostgreSQL, for working without Neon | `--profile local-db` |

All three Python services use **one image**: the same code, with a different command each.

**Health checks:** Docker asks the API (`/health`) and the dashboard (`/_stcore/health`) every
5 seconds whether they are fine. The dashboard and the feed only start once the API answers. A
crashed service is restarted (`restart: unless-stopped`).

---

## 3. Design decisions

| Decision | Why |
|---|---|
| One image for the API, the dashboard and the feed | One build; the three parts can never run different versions of the code |
| Libraries installed **before** the code is copied | Docker reuses the slow library step (about 2 minutes) when only the code changes |
| **PyTorch from its CPU-only package index**, same version (2.15.0) | The default package adds gigabytes of GPU libraries that the CPU-only models never use |
| **Data, models and results are not copied into the image**; they are attached read-only when the containers start | They are made by scripts 01–06 (2.7 GB of data); the image stays code-only, and a re-trained model needs no rebuild |
| **The database password is never in the image**: `.env` is excluded from the build, and `DATABASE_URL` is handed to the containers at start-up | An image can be shared without sharing the secret (checked: no `.env` and no `DATABASE_URL` inside) |
| The containers run as a normal user (`app`), not as root | If a service were attacked, it would have fewer rights |
| Host ports can be changed (`API_PORT`, `DASHBOARD_PORT`) | Starts even when 8000 or 8501 are already in use |
| The local database is optional (profile `local-db`), and its port is open only on this computer | Neon is the default; the local one is for working offline |

---

## 4. Checks

| Check | Result |
|---|---|
| Image build (Apple M2) | 2.5 minutes; image **2.36 GB** (mostly PyTorch and the data libraries) |
| Inside the image | torch 2.15.0+cpu, scikit-learn 1.9.1, pandas 3.1.1, streamlit 1.67.0 (as `requirements.txt`); user `app`; no `.env`; no `DATABASE_URL` |
| `docker-compose up` | API and dashboard **healthy after 12 s**; the API's log says "alert store ready" (Neon) |
| Log feed in a container (12 BGL log hours) | The same result as outside Docker: 17 windows judged, the FATAL burst at 08:40 caught, stored in Neon |
| **End to end through the Docker API**: both test periods replayed, results read back from Neon (`scripts/11_check_replay.py --api ...`) | **0 alert differences** vs the offline evaluation: BGL 2,899 of 2,899, HDFS 115,012 of 115,012 ([`replay_check_docker.json`](../reports/results/replay_check_docker.json)) |
| Dashboard in Docker | All tabs open without errors; it reads Neon and the mounted labels, and reaches the API at `http://api:8000` |
| Without Neon (`--profile local-db`) | The API stores into the local `db` container; the 12-hour BGL replay was stored there |

**The containers give exactly the evaluated decisions**, even though they run on Linux with a
different build of Python and PyTorch. HDFS scores differ by at most 1.3 × 10⁻⁷, the float32
rounding explained in docs/06 §6.

**Speed:** the end-to-end check reached about 40,000 lines/s (BGL) and 50,000 lines/s (HDFS) through the
Docker API. Without Docker it was about 70,000–75,000. On a Mac, Docker runs inside a small Linux
virtual machine, which costs some speed. That's still far more than the logs need: HDFS writes
about 170 lines per second in its test period.

---

## 5. Limitations

- **The image is large (2.36 GB).** A slimmer image is possible: leave out the libraries only the
  offline scripts use, or remove PyTorch if DeepLog is never served. Not worth the risk of breaking
  the identical-versions guarantee here.
- **Data and models must exist first** (scripts 01–06). A production setup would download the
  trained models from a model registry instead.
- **Single machine:** Compose runs everything on one computer. A real deployment would use an
  orchestrator such as Kubernetes, with several API copies. The API keeps its open groups in memory,
  so that would need more design.
- **No HTTPS or login** in front of the API and the dashboard (local demo).

---

## 6. Questions a lecturer might ask

**What is the difference between an image and a container?** The image is the packed, read-only
recipe result; a container is a running copy of it. One image runs here as three containers.

**Why not copy the models and data into the image?** The data is 2.7 GB and is re-made by the
scripts. Keeping it outside makes the image code-only, and a re-trained model only needs a restart,
not a rebuild.

**How do you keep the database password out of Docker?** `.env` is listed in `.dockerignore`, so it
never enters the image. The password reaches the containers only when they start, as the
`DATABASE_URL` environment variable. I checked that the image contains neither.

**How do you know the containers behave like your evaluated system?** Both test periods were replayed
through the API running in Docker, and the stored results were compared with the offline evaluation:
0 differences.

**Why is PyTorch installed from a different index?** The normal PyTorch package includes GPU
(CUDA) libraries of several GB. The models run on CPU only, so the CPU build (same version) is used.
