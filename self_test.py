from app.airwrite import AirWritingRecognizer, _templates
from app.expression import evaluate_tokens
from app.session import SessionController


def test_recognizer_templates():
    r = AirWritingRecognizer()
    for char, points in _templates().items():
        expected = "NUMBER" if char.isdigit() else "OPERATOR"
        predicted, score = r.predict(points, expected=expected)
        assert predicted == char, (char, predicted, score)


def test_token_semantics():
    s = SessionController()
    s.start()
    s.identity_ok(1.0, enrolled_now=True)
    s.state = s.state.READY

    for c in "156":
        s.add_character(c, 1.0)

    assert s.buffer_chars == ["1", "5", "6"]
    s.commit_token()
    assert s.tokens == ["156"]

    s.add_character("+", 1.0)
    s.commit_token()
    assert s.tokens == ["156", "+"]

    s.undo()
    assert s.tokens == ["156"]

    s.add_character("+", 1.0)
    s.commit_token()
    for c in "42":
        s.add_character(c, 1.0)
    s.commit_token()
    s.add_character("=", 1.0)
    s.commit_token()

    assert s.tokens == ["156", "+", "42", "="]
    assert s.result == "198"


def test_safe_expression():
    expr, value = evaluate_tokens(["156", "+", "42", "="])
    assert value == 198
    assert expr == "156+42"

def test_operator_gesture_tokens():
    s = SessionController()
    s.start()
    s.identity_ok(1.0, enrolled_now=True)
    s.state = s.state.READY

    for c in "156":
        s.add_character(c, 1.0)

    # Operator gesture also safely commits the buffered number first.
    s.apply_operator("+")
    assert s.tokens == ["156", "+"]

    for c in "42":
        s.add_character(c, 1.0)

    s.apply_operator("=")
    assert s.tokens == ["156", "+", "42", "="]
    assert s.result == "198"

if __name__ == "__main__":
    test_recognizer_templates()
    test_token_semantics()
    test_safe_expression()
    test_operator_gesture_tokens()
    print("SELF TEST OK")
