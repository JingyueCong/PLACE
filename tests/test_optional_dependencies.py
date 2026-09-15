import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def top_level_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)
    return imports


def test_recap_training_path_does_not_eagerly_import_deepspeed():
    imports = top_level_imports(ROOT / "ULD" / "uld" / "hfutil" / "hf_trainers.py")
    assert "deepspeed" not in imports


def test_eval_seed_helper_does_not_import_trainer_registry():
    source = (ROOT / "open-unlearning" / "src" / "eval.py").read_text(encoding="utf-8")
    assert "from seed_utils import seed_everything" in source
    assert "from trainer.utils import seed_everything" not in source
