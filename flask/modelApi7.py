from flask import Flask, request, jsonify
import joblib
import numpy as np
import tensorflow as tf
import torch
import os

from tensorflow.keras.models import load_model
from model7.preprocess7 import preprocess_raw_input_v2

# env/Scripts/activate
# ------------ 디버그 설정 ------------ #
DEBUG = os.getenv("DEBUG", "false").lower() == "true"

def debug_print(*args):
    if DEBUG:
        print(*args)

# ------------ Flask App ------------ #
app = Flask(__name__)

# ------------ 커스텀 손실 함수 ------------ #
@tf.keras.utils.register_keras_serializable()
def loss_fn(y_true, y_pred):
    y_true = tf.cast(y_true, tf.int32)
    y_true_onehot = tf.one_hot(y_true, depth=tf.shape(y_pred)[-1])
    cross_entropy = -y_true_onehot * tf.math.log(tf.clip_by_value(y_pred, 1e-9, 1.0))
    weight = 0.25 * tf.pow(1 - y_pred, 2.0)
    loss = weight * cross_entropy
    return tf.reduce_sum(loss, axis=1)

# ------------ 리소스 로딩 ------------ #
debug_print("🔄 전처리 리소스 로딩 중...")
feature_order = joblib.load('model7/feature_order.pkl')
label_encoders = joblib.load('model7/label_encoders.pkl')
imputer = joblib.load('model7/imputer.pkl')
scaler = joblib.load('model7/scaler.pkl')
class_names = joblib.load('model7/class_names.pkl')
damage_lookup = joblib.load('model7/damage_lookup.pkl')
debug_print("✅ 전처리 리소스 로딩 완료")

# ------------ 위험도 관련 ------------ #
class_risk_weights = {
    '차량/기타설비 I/F': 3,
    '기타': 2,
    '선로및구조물': 2,
    '환경요인': 2,
    '차량/선로 및 구조물 I/F': 1,
    '차량/통신 I/F': 1,
    '차량/신호 I/F': 1,
    '정보통신설비': 1
}

def damage_to_risk(damage):
    if damage == 0:
        return 1
    elif damage <= 0.2:
        return 2
    elif damage <= 1.0:
        return 3
    elif damage <= 5.0:
        return 4
    else:
        return 5

# ------------ 모델 로딩 ------------ #
debug_print("🔄 Keras 모델 로딩 중...")
model = load_model('model7/railway_safety_model.h5', custom_objects={"loss_fn": loss_fn})
debug_print("✅ 모델 로딩 완료")

# ------------ 예측 API ------------ #
@app.route('/predict_batch_v3', methods=['POST'])
def predict_batch_v3():
    try:
        input_list = request.get_json()
        debug_print(f"📥 [입력 수신] 요청 샘플 수: {len(input_list)}")

        # [1] 배치 전처리
        input_tensors = []
        for idx, input_dict in enumerate(input_list):
            debug_print(f"\n🧾 [샘플 {idx}] 원본 입력 데이터:")
            debug_print(input_dict)

            input_tensor = preprocess_raw_input_v2(
                raw_input=input_dict,
                feature_order=feature_order,
                label_encoders=label_encoders,
                imputer=imputer,
                scaler=scaler
            )
            input_tensor = input_tensor.squeeze(0)  # 👈 여기 추가!
            input_tensors.append(input_tensor)

        # [2] 배치 모델 추론
        batch_tensor = torch.stack(input_tensors)
        debug_print(f"📡 배치 모델 추론 시작... (배치 크기: {batch_tensor.shape[0]})")
        preds = model.predict(batch_tensor.numpy())
        debug_print(f"📈 전체 softmax 출력 예시 [0]: {preds[0].tolist()}")

        # [3] 결과 생성
        results = []
        for idx, (input_dict, pred_row) in enumerate(zip(input_list, preds)):
            pred_class_idx = np.argmax(pred_row)
            pred_class_name = class_names[pred_class_idx]
            
            #⛔️ 예측 결과가 "근본원인"일 경우 → "없음"으로 치환
            if pred_class_name == "근본원인":
                debug_print(f"⚠️ 예측된 사고유형이 '근본원인' → '없음'으로 치환합니다.")
                pred_class_name = "외적 요인"
            
            debug_print(f"🎯 [샘플 {idx}] 예측 결과: {pred_class_name} (index={pred_class_idx})")

            rail_type = input_dict.get("철도구분", "").strip()
            line = input_dict.get("노선", "").strip()
            key = (str(rail_type), str(pred_class_name), str(line))
            damage = damage_lookup.get(key, 0)
            damage_risk = damage_to_risk(damage)
            class_risk = class_risk_weights.get(pred_class_name, 0)
            combined_risk = round(0.6 * damage_risk + 0.4 * class_risk, 2)

            debug_print(f"🧮 [샘플 {idx}] 피해액: {damage}, 피해위험도: {damage_risk}, 유형위험도: {class_risk}, 복합위험도: {combined_risk}")

            results.append({
                "predicted_cause": pred_class_name,
                "damage_risk": damage_risk,
                "class_risk": class_risk,
                "combined_risk": combined_risk
            })

        debug_print("✅ 전체 예측 완료. 총 결과 수:", len(results))
        return jsonify(results)

    except Exception as e:
        debug_print("❌ 예측 중 오류 발생:", str(e))
        return jsonify({"error": str(e)}), 500

# ------------ 서버 실행 ------------ #
if __name__ == '__main__':
    debug_print("🚀 Flask 추론 서버 실행 시작")
    app.run(debug=True)
