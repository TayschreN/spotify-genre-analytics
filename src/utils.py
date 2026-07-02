import duckdb
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
DB_PATH = ROOT / "data" / "spotify.duckdb"
MODELS_DIR = ROOT / "models"

DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def get_connection():
    """Retorna conexão persistente com o DuckDB local."""
    return duckdb.connect(str(DB_PATH))


# Mapeamento de redução dos 125 gêneros originais para macro-categorias.
# Ajuste livremente depois de olhar a lista real de track_genre no seu CSV.
GENRE_MAP = {
    "metal": ["death-metal", "black-metal", "metalcore", "heavy-metal", "metal", "grindcore",
              "industrial", "goth", "hardcore"],
    "rock": ["rock", "alt-rock", "hard-rock", "punk-rock", "psych-rock", "rock-n-roll", "grunge",
             "alternative", "garage", "rockabilly", "punk", "emo", "j-rock"],
    "pop": ["pop", "power-pop", "indie-pop", "synth-pop", "k-pop", "j-pop", "pop-film",
            "indie", "cantopop", "mandopop", "j-idol", "disney", "show-tunes"],
    "electronic": ["edm", "house", "techno", "trance", "dubstep", "drum-and-bass", "electronic",
                   "electro", "minimal-techno", "detroit-techno", "chicago-house", "deep-house",
                   "progressive-house", "club", "dance", "breakbeat", "idm", "hardstyle",
                   "j-dance", "garage", "trip-hop"],
    "hip_hop": ["hip-hop", "trap", "r-n-b"],
    "classical": ["classical", "opera", "piano"],
    "jazz_blues": ["jazz", "blues", "soul", "funk", "groove"],
    "latin": ["salsa", "samba", "reggaeton", "latin", "latino", "mpb", "pagode", "sertanejo",
              "forro", "tango"],
    "folk_acoustic": ["folk", "acoustic", "singer-songwriter", "country", "bluegrass",
                       "honky-tonk", "guitar"],
    "world": ["indian", "iranian", "turkish", "brazil", "french", "german", "swedish", "spanish",
              "british", "afrobeat", "malay", "world-music", "reggae", "dancehall", "dub", "ska",
              "gospel"],
    "ambient_chill": ["ambient", "chill", "study", "sleep", "new-age", "romance", "sad", "happy"],
    "novelty": ["anime", "children", "kids", "comedy", "party"],
}


def map_genre(genre: str) -> str:
    """Mapeia um track_genre original para a macro-categoria. Retorna 'other' se não encontrado."""
    for macro, genres in GENRE_MAP.items():
        if genre in genres:
            return macro
    return "other"