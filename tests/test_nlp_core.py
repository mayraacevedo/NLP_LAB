"""Pruebas sin descargar modelos: stubs que imitan la API de Transformers."""

import unittest

from nlp_core import analizar_frase, procesar_puntuaciones


class TokenFalso:
    def __init__(self, texto):
        self.text = texto
        self.lemma_ = texto.lower()
        self.is_stop = texto.lower() in {"me", "la", "no"}
        self.is_punct = False


def nlp_falso(texto):
    return [TokenFalso(t) for t in texto.split()]


class TokenizadorFalso:
    def __call__(self, texto, truncation=False, max_length=None, add_special_tokens=True):
        tokens = [101] + list(range(102, 102 + len(texto.split()))) + [102]
        if truncation and max_length is not None and len(tokens) > max_length:
            tokens = tokens[:max_length - 1] + [102]
        return {"input_ids": tokens}

    def convert_ids_to_tokens(self, ids):
        return ["[CLS]" if x == 101 else "[SEP]" if x == 102 else f"t{x}" for x in ids]


class ClasificadorFalso:
    tokenizer = TokenizadorFalso()

    def __init__(self):
        self.ultima_llamada = None

    def __call__(self, texto, **kwargs):
        self.ultima_llamada = (texto, kwargs)
        return [
            {"label": "4 stars", "score": 0.468},
            {"label": "1 star", "score": 0.032},
            {"label": "2 stars", "score": 0.1},
            {"label": "3 stars", "score": 0.15},
            {"label": "5 stars", "score": 0.25},
        ]


class TestNlpCore(unittest.TestCase):
    def test_cinco_clases_ordenadas_y_suman_uno(self):
        salida = analizar_frase("me gusta la clase", nlp_falso, ClasificadorFalso())
        self.assertEqual(salida["etiqueta"], "4 stars")
        self.assertAlmostEqual(salida["puntuacion"], 0.468)
        self.assertEqual([x["estrellas"] for x in salida["distribucion"]], [1, 2, 3, 4, 5])
        self.assertAlmostEqual(sum(x["puntuacion"] for x in salida["distribucion"]), 1.0)

    def test_pide_top_k_none_para_cinco_resultados(self):
        clasificador = ClasificadorFalso()
        analizar_frase("me gusta", nlp_falso, clasificador)
        self.assertIsNone(clasificador.ultima_llamada[1]["top_k"])
        self.assertTrue(clasificador.ultima_llamada[1]["truncation"])

    def test_tokens_y_truncamiento(self):
        salida = analizar_frase("uno dos tres cuatro", nlp_falso, ClasificadorFalso(), 4)
        self.assertTrue(salida["truncado"])
        self.assertEqual(salida["total_modelo"], 4)
        self.assertEqual(salida["total_original_modelo"], 6)

    def test_no_elimina_negaciones(self):
        clasificador = ClasificadorFalso()
        analizar_frase("No me gusta", nlp_falso, clasificador)
        self.assertEqual(clasificador.ultima_llamada[0], "No me gusta")

    def test_frase_vacia(self):
        with self.assertRaises(ValueError):
            analizar_frase("  ", nlp_falso, ClasificadorFalso())

    def test_max_length_invalido(self):
        with self.assertRaises(ValueError):
            analizar_frase("hola", nlp_falso, ClasificadorFalso(), 2)

    def test_rechaza_etiquetas_faltantes(self):
        with self.assertRaises(ValueError):
            procesar_puntuaciones([{"label": "4 stars", "score": 0.5}])

    def test_rechaza_etiquetas_repetidas(self):
        with self.assertRaises(ValueError):
            procesar_puntuaciones([{"label": "1 star", "score": 0.2}] * 5)

    def test_tolera_formato_anidado(self):
        resultados = ClasificadorFalso()("hola")
        self.assertEqual(len(procesar_puntuaciones([resultados])), 5)


if __name__ == "__main__":
    unittest.main()
