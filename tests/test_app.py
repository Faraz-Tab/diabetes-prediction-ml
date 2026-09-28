def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.get_json()["status"] == "ok"


def test_home_renders_form_and_disclaimer(client):
    body = client.get("/").get_data(as_text=True)
    assert 'name="glucose"' in body
    assert "not a medical device" in body


def test_json_prediction(client, valid):
    response = client.post("/predict", json=valid)
    data = response.get_json()
    assert response.status_code == 200
    assert data["prediction"] in (0, 1)
    assert 0.0 <= data["probability"] <= 1.0


def test_form_prediction(client, valid):
    response = client.post("/predict", data=valid)
    assert response.status_code == 200
    assert "risk of diabetes" in response.get_data(as_text=True)


def test_missing_and_non_numeric_fields_rejected(client, valid):
    del valid["age"]
    valid["glucose"] = "abc"
    response = client.post("/predict", json=valid)
    errors = response.get_json()["errors"]
    assert response.status_code == 400
    assert errors == {"glucose": "must be a number", "age": "required"}


def test_out_of_range_and_fractional_int_rejected(client, valid):
    valid["bmi"] = "500"
    valid["pregnancies"] = "2.5"
    response = client.post("/predict", data=valid)
    assert response.status_code == 400
    body = response.get_data(as_text=True)
    assert "must be between 0 and 80" in body
    assert "must be a whole number" in body


def test_unknown_route_is_404(client):
    assert client.get("/nope").status_code == 404
