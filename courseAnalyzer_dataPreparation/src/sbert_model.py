"""SBERT encoder and reference prototypes from the legacy analysis job."""
import numpy as np
import os

POSITIVE = ["excellent course", "great learning", "amazing content"]
NEGATIVE = ["bad course", "poor explanation", "waste of time"]
NEUTRAL = ["average course", "okay content"]
LABELS = ("positive", "negative", "neutral")


class ReferenceSBERT:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", device=None, batch_size=64,
                 model_dir=None, reference_embeddings_path=None, require_local=False):
        from pathlib import Path
        use_local = bool(model_dir and Path(model_dir).is_dir())
        if require_local and not use_local:
            raise FileNotFoundError(f"Persisted SBERT model is missing at {model_dir}; restore the local model artifact before inference.")
        local_path = str(model_dir) if use_local else model_name
        options = {"device": device} if device else {}
        if use_local:
            options.update(local_files_only=True, model_kwargs={"local_files_only": True},
                           processor_kwargs={"local_files_only": True},
                           config_kwargs={"local_files_only": True})
        elif model_dir and not require_local:
            # Resolve a cached snapshot directly so offline mode is active before
            # importing transformers/huggingface_hub (which cache env settings).
            cache_root = os.environ.get("HF_HUB_CACHE")
            if not cache_root:
                cache_root = str(Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub")
            repo_cache = Path(cache_root) / ("models--" + model_name.replace("/", "--"))
            reference = repo_cache / "refs/main"
            snapshot = repo_cache / "snapshots" / (reference.read_text(encoding="utf-8").strip() if reference.is_file() else "")
            if snapshot.is_dir():
                local_path = str(snapshot)
                options.update(local_files_only=True, model_kwargs={"local_files_only": True},
                               processor_kwargs={"local_files_only": True},
                               config_kwargs={"local_files_only": True})
        offline_values = {name: os.environ.get(name) for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")}
        if options.get("local_files_only"):
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            # Some Hugging Face modules may have imported constants before these
            # environment flags were set. Keep their runtime offline switch aligned.
            try:
                import huggingface_hub.constants as hub_constants
                hub_constants.HF_HUB_OFFLINE = True
            except ImportError:
                pass
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(local_path, **options)
        finally:
            for name, value in offline_values.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
        self.batch_size = batch_size
        if reference_embeddings_path and Path(reference_embeddings_path).is_file():
            refs = np.load(reference_embeddings_path)
            if refs.shape != (8, 384):
                refs = self.model.encode(POSITIVE + NEGATIVE + NEUTRAL, batch_size=batch_size,
                                         convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
        else:
            refs = self.model.encode(POSITIVE + NEGATIVE + NEUTRAL, batch_size=batch_size,
                                     convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
        self.references = (refs[:3], refs[3:6], refs[6:])

    def save_model(self, path):
        self.model.save(str(path))

    def save_reference_embeddings(self, path):
        refs = np.vstack(self.references).astype(np.float32)
        path = __import__("pathlib").Path(path)
        temporary = path.with_name(path.stem + ".tmp.npy")
        np.save(temporary, refs)
        temporary.replace(path)

    def embed(self, texts):
        return self.model.encode(texts, batch_size=self.batch_size, convert_to_numpy=True,
                                 normalize_embeddings=True, show_progress_bar=False)

    def classify_embeddings(self, vectors, uncertainty_margin=0.03):
        scores = np.column_stack([np.max(vectors @ refs.T, axis=1) for refs in self.references])
        order = np.argsort(scores, axis=1)
        winners = order[:, -1]
        margins = scores[np.arange(len(scores)), order[:, -1]] - scores[np.arange(len(scores)), order[:, -2]]
        return scores, np.asarray(LABELS)[winners], margins, margins < uncertainty_margin

    def classify(self, texts, uncertainty_margin=0.03):
        return self.classify_embeddings(self.embed(texts), uncertainty_margin)
