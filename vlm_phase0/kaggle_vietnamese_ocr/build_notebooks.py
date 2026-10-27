"""Generate standalone Kaggle notebooks from audited source files."""
from __future__ import annotations

import json
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source.splitlines(keepends=True)}


def embed(*names: str) -> dict:
    files = {name: (ROOT / name).read_text(encoding="utf-8") for name in names}
    source = "from pathlib import Path\nimport sys\nLIB = Path('/kaggle/working/ocr_benchmark_lib')\nLIB.mkdir(parents=True, exist_ok=True)\n"
    for name, payload in files.items():
        source += f"(LIB / {name!r}).write_text({payload!r}, encoding='utf-8')\n"
    source += "sys.path.insert(0, str(LIB))\nprint('Embedded benchmark source ready:', sorted(p.name for p in LIB.iterdir()))"
    return code(source)


def save(name: str, cells: list[dict], gpu: bool) -> None:
    for index, cell in enumerate(cells):
        payload = f"{name}:{index}:{''.join(cell['source'])}".encode("utf-8")
        cell["id"] = hashlib.sha256(payload).hexdigest()[:12]
    notebook = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                "language_info": {"name": "python"}, "kaggle": {"accelerator": "gpu" if gpu else "none", "internet": True}},
                "nbformat": 4, "nbformat_minor": 5}
    (ROOT / name).write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")


def inference_notebook(model: str) -> None:
    is_paddle = model == "paddle"
    label = "PaddleOCR-VL 1.6" if is_paddle else "DeepSeek-OCR 2"
    name = "01_paddleocr_vl_1_6_kaggle.ipynb" if is_paddle else "02_deepseek_ocr_2_kaggle.ipynb"
    cells = [
        md(f"# {label} — CC-OCR Vietnamese trên Kaggle\n\nBenchmark zero-shot hệ OCR đầu cuối. Notebook này tải đúng checkpoint chính thức và ghi từng ảnh để có thể resume. `SMOKE_TEST` gồm 5 ảnh đầu theo thứ tự TSV; `FULL_BENCHMARK` gồm mọi ảnh hợp lệ. Không suy diễn kết quả trước khi chạy GPU thực tế."),
        md("## 1. Kiểm tra môi trường trước cài đặt\n\nĐọc Python, Torch, CUDA, driver, CC, VRAM, RAM, đĩa và internet. Dừng nếu `cuda:0` không phải T4. Hai T4 có VRAM riêng; notebook chỉ dùng GPU 0."),
        embed("benchmark_core.py", "model_adapters.py"),
        code("from pathlib import Path\nfrom benchmark_core import environment, require_t4, write_json\nROOT = Path('/kaggle/working/vietnamese_ocr')\nROOT.mkdir(parents=True, exist_ok=True)\nENV = environment()\nwrite_json(ROOT / 'environment_preinstall.json', ENV)\nprint(ENV)\nrequire_t4(ENV)\nif not ENV['internet']: raise RuntimeError('Kaggle Internet chưa bật')"),
        md("## 2. Dependencies được pin và kiểm tra lại\n\nGiữ Torch hiện có của Kaggle; không nâng cấp toàn bộ môi trường. Paddle dùng wheel CUDA 12.6 đã tài liệu hóa. DeepSeek dùng Transformers 4.46.3 theo model card. Nếu CUDA hoặc ABI khác, dừng và điều tra thay vì tự đổi phiên bản."),
        code("from model_adapters import install_common, install_paddle\ninstall_common()\n" + ("install_paddle(ENV)\n" if is_paddle else "import subprocess, sys\nsubprocess.check_call([sys.executable, '-m', 'pip', 'install', 'transformers==4.46.3', 'tokenizers==0.20.3', 'accelerate==1.0.1', 'einops==0.8.1', 'addict==2.4.0', 'easydict==1.13'])\n") + "import importlib.metadata as im\nprint({p: im.version(p) for p in ['torch','huggingface_hub','rapidfuzz','pandas']})"),
        md("## 3. Dataset: schema, ảnh và nhãn\n\nTải TSV bằng `hf_hub_download` theo commit SHA. Đầu tiên in tên cột và bản xem trước không chứa Base64. Sau đó giải mã, xác thực ảnh, ghi `manifest.csv` và mọi mẫu lỗi. `answer` chỉ được dùng làm ground truth sau khi người chạy xem ảnh và xác nhận ý nghĩa của cột; không đưa nó vào prompt."),
        code("from benchmark_core import download_tsv, inspect_tsv, make_manifest, manifest_fingerprint, sha256\nfrom IPython.display import display\nfrom PIL import Image\nimport pandas as pd\nDATASET_REVISION_OVERRIDE = None  # Set to the other notebook's commit SHA when reproducing\nTSV, DATASET_REVISION = download_tsv(ROOT, DATASET_REVISION_OVERRIDE)\nSCHEMA = inspect_tsv(TSV)\nwrite_json(ROOT / 'dataset_schema.json', SCHEMA)\nprint('Columns:', SCHEMA['columns'], 'Dataset revision:', DATASET_REVISION)\ndisplay(pd.DataFrame(SCHEMA['preview']))\nMANIFEST, REJECTED = make_manifest(TSV, ROOT / 'dataset')\nprint('Valid:', len(MANIFEST), 'Rejected:', len(REJECTED), 'Duplicate images:', sum(bool(x['duplicate_of']) for x in MANIFEST))\nfor row in MANIFEST[:5]:\n    print('ID:', row['sample_id'], 'Question:', row['question'], 'Answer:', row['ground_truth'])\n    display(Image.open(row['image_path']))\nprint('Inspect the five images above before confirming answer semantics.')"),
        code("CONFIRM_GROUND_TRUTH_SEMANTICS = False  # Set True only after visual and question/answer review\nassert CONFIRM_GROUND_TRUTH_SEMANTICS, 'Confirm that answer is the literal image transcription'\nMODE = 'SMOKE_TEST'  # After inspecting smoke outputs, change to FULL_BENCHMARK and rerun\nassert MODE in {'SMOKE_TEST','FULL_BENCHMARK'}\nimport random, numpy as np, torch\nrandom.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)\nOUT = ROOT / ('paddle' if " + str(is_paddle) + " else 'deepseek') / MODE.lower()\nOUT.mkdir(parents=True, exist_ok=True)\nimport shutil\nshutil.copy2(ROOT / 'dataset' / 'manifest.csv', OUT / 'manifest.csv')\nshutil.copy2(ROOT / 'dataset' / 'rejected_samples.csv', OUT / 'rejected_samples.csv')\nshutil.copytree(ROOT / 'dataset' / 'images', OUT / 'images', dirs_exist_ok=True)"),
        md("## 4. Nạp checkpoint chính thức và kiểm chứng adapter\n\nPaddle chạy pipeline `PaddleOCRVL(pipeline_version='v1.6')` có layout analysis. DeepSeek dùng checkpoint `deepseek-ai/DeepSeek-OCR-2`, eager attention và FP16 trên T4. Bản vá DeepSeek được hash và ghi rõ trong `run_manifest.json`; FP16 có thể làm thay đổi độ chính xác so với BF16 chính thức."),
        code(("from model_adapters import PaddleAdapter, PADDLE_ID, MAX_NEW_TOKENS\nfrom huggingface_hub import HfApi\nMODEL_REVISION = HfApi().model_info(PADDLE_ID).sha\nADAPTER = PaddleAdapter()\nMODEL_ID = PADDLE_ID\nPATCH = None\n" if is_paddle else "from model_adapters import DeepSeekAdapter, DEEPSEEK_ID, DEEPSEEK_PROMPT, MAX_NEW_TOKENS\nMODEL_REVISION_OVERRIDE = None  # Pin a prior run's commit SHA if needed\nADAPTER = DeepSeekAdapter(ROOT, MODEL_REVISION_OVERRIDE)\nMODEL_REVISION = ADAPTER.revision\nMODEL_ID = DEEPSEEK_ID\nPATCH = ADAPTER.patch\n") + "from benchmark_core import manifest_fingerprint\nRUN_META = {'model_id': MODEL_ID, 'model_revision': MODEL_REVISION, 'dataset_id': 'wulipc/CC-OCR', 'dataset_file': 'multi_lan_ocr/Vietnamese/Vietnamese_150.tsv', 'dataset_revision': DATASET_REVISION, 'dataset_sha256': SCHEMA['tsv_sha256'], 'manifest_sha256': manifest_fingerprint(MANIFEST), 'valid_sample_count': len(MANIFEST), 'rejected_sample_count': len(REJECTED), 'mode': MODE, 'batch_size': 1, 'seed': 42, 'max_new_tokens': MAX_NEW_TOKENS, 'device': 'cuda:0', 'initialization_seconds': ADAPTER.init_seconds, 'patch': PATCH, 'precision': 'Paddle default' if " + str(is_paddle) + " else 'FP16', 'preprocessing': 'official PaddleOCRVL pipeline' if " + str(is_paddle) + " else 'DeepSeek official dynamic crop 1024/768', 'normalization': ['strict','normalized NFC whitespace'], 'ground_truth_confirmed': CONFIRM_GROUND_TRUTH_SEMANTICS}\nprevious_path = OUT / 'run_manifest.json'\nif previous_path.exists():\n    import json\n    previous = json.loads(previous_path.read_text(encoding='utf-8'))\n    for key in ['model_id','model_revision','dataset_revision','dataset_sha256','manifest_sha256','mode','precision','patch','max_new_tokens']:\n        if previous[key] != RUN_META[key]: raise RuntimeError(f'Resume config mismatch: {key}')\nelse:\n    write_json(previous_path, RUN_META)\nwrite_json(OUT / 'benchmark_config.json', RUN_META)\nwrite_json(OUT / 'environment.json', environment())\nprint(RUN_META)"),
        md("## 5. Warm-up, smoke test và benchmark\n\nWarm-up dùng ảnh đầu tiên, không tính trong latency chính. Mỗi ảnh đồng bộ CUDA trước/sau đo, ghi checkpoint JSONL ngay sau inference. `timeout` là ngưỡng hậu kiểm, không cắt ngang lệnh GPU đang treo. OOM dừng sau một mẫu để tránh lặp lỗi. Kết quả lỗi được giữ lại và tính như chuỗi rỗng trong CER/WER đầu cuối."),
        code("from benchmark_core import run_benchmark\nROWS = run_benchmark(MANIFEST, OUT, ADAPTER.infer, ADAPTER.sync, ADAPTER.gpu_peak, MODE, Path(MANIFEST[0]['image_path']), reset_peak=ADAPTER.reset_peak)\nprint('Recorded:', len(ROWS), '/', 5 if MODE == 'SMOKE_TEST' else len(MANIFEST))\ndisplay(pd.DataFrame(ROWS)[['sample_id','status','latency_s','error']].head(10))"),
        md("## 6. Xem ảnh và xuất kết quả\n\nKiểm tra output thô cùng ảnh và nhãn. `metrics.json` chứa cả strict và normalized; đây là kết quả của chế độ hiện tại, không ghép smoke với full. Lưu version của notebook để tải toàn bộ `/kaggle/working/vietnamese_ocr` từ Kaggle Notebook Outputs."),
        code("from benchmark_core import score\nfor row in ROWS[:5]:\n    match = next(x for x in MANIFEST if x['sample_id'] == row['sample_id'])\n    display(Image.open(match['image_path']))\n    print('GT:', row['ground_truth'])\n    print('Prediction:', row['prediction_raw'])\n    print('Status:', row['status'], 'Error:', row['error'])\nprint('Strict:', score(ROWS, 'strict'))\nprint('Normalized:', score(ROWS, 'normalized'))\nprint('Outputs:', sorted(p.name for p in OUT.iterdir()))"),
    ]
    save(name, cells, gpu=True)


def compare_notebook() -> None:
    cells = [
        md("# So sánh OCR tiếng Việt — CPU\n\nChỉ đọc hai output `FULL_BENCHMARK`. Ghép theo `(sample_id, image_sha256)` và chặn nếu revision, manifest, nhãn hoặc tập mẫu khác. Không tạo số đo giả."),
        embed("benchmark_core.py", "compare.py"),
        md("## 1. Cài thư viện phân tích và tìm Kaggle Inputs\n\nThêm hai Notebook Outputs hoặc Kaggle Datasets làm inputs. Notebook tự tìm đúng `run_manifest.json` theo model ID. Nếu nhiều bản, đặt đường dẫn tường minh bên dưới."),
        code("import subprocess, sys\nsubprocess.check_call([sys.executable, '-m','pip','install','rapidfuzz==3.14.1','numpy==2.0.2','matplotlib==3.10.3'])\nfrom pathlib import Path\nimport json\nROOT = Path('/kaggle/working/vietnamese_ocr_compare')\nROOT.mkdir(parents=True, exist_ok=True)\nPADDLE_OUTPUT_DIR = None  # e.g. Path('/kaggle/input/<notebook-output>/.../full_benchmark')\nDEEPSEEK_OUTPUT_DIR = None\nfound = []\nfor p in Path('/kaggle/input').rglob('run_manifest.json'):\n    try:\n        meta = json.loads(p.read_text(encoding='utf-8'))\n        found.append((p.parent, meta.get('model_id'), meta.get('mode')))\n    except (OSError, ValueError): pass\nprint('Found:', found)\ndef choose(model_id):\n    dirs = [d for d,m,mode in found if m == model_id and mode == 'FULL_BENCHMARK']\n    if len(dirs) != 1: raise RuntimeError(f'Expected exactly one full output for {model_id}; found {dirs}')\n    return dirs[0]\nPADDLE_OUTPUT_DIR = Path(PADDLE_OUTPUT_DIR) if PADDLE_OUTPUT_DIR else choose('PaddlePaddle/PaddleOCR-VL-1.6')\nDEEPSEEK_OUTPUT_DIR = Path(DEEPSEEK_OUTPUT_DIR) if DEEPSEEK_OUTPUT_DIR else choose('deepseek-ai/DeepSeek-OCR-2')"),
        md("## 2. Kiểm tra tính công bằng và tính metrics\n\nStrict giữ nguyên ký tự. Normalized chỉ NFC và khoảng trắng. CER là edit distance ở mức ký tự, không cắt tại 1. WER là token theo khoảng trắng, không phải tách từ tiếng Việt ngôn ngữ học. Runtime error/OOM/timeout được báo riêng và tính dự đoán rỗng cho chất lượng đầu cuối."),
        code("from compare import compare\nfrom benchmark_core import write_json\ntry:\n    SUMMARY = compare(PADDLE_OUTPUT_DIR, DEEPSEEK_OUTPUT_DIR, ROOT)\nexcept Exception as exc:\n    write_json(ROOT / 'comparison_summary.json', {'status': 'NOT_EVALUATED', 'reason': str(exc)})\n    (ROOT / 'comparison_report.md').write_text('# Chưa thể so sánh\\n\\nNOT_EVALUATED: ' + str(exc), encoding='utf-8')\n    raise\nfrom IPython.display import display\nimport pandas as pd\ndisplay(pd.DataFrame(SUMMARY['table']))"),
        md("## 3. Biểu đồ và mười ca lỗi\n\nCác ảnh lỗi chọn bằng quy tắc cố định: CER của mô hình tốt hơn trên từng ảnh cao nhất, sau đó theo sample ID. Nhãn lỗi là heuristic dựa trên edit; phải đọc diff để kiểm chứng. HTML tự chứa ảnh nếu Notebook Output có `manifest.csv` và ảnh."),
        code("from IPython.display import Image, display, HTML\nfor name in ['quality.png','cer_boxplot.png','latency_histogram.png','cer_vs_latency.png','vram.png','wins.png']:\n    display(Image(filename=str(ROOT / name)))\nprint('Report:', ROOT / 'comparison_report.md')\nprint('Error analysis:', ROOT / 'error_analysis.html')\ndisplay(HTML((ROOT / 'error_analysis.html').read_text(encoding='utf-8')))"),
    ]
    save("03_compare_results.ipynb", cells, gpu=False)


if __name__ == "__main__":
    inference_notebook("paddle")
    inference_notebook("deepseek")
    compare_notebook()
    print("Generated three standalone notebooks")
