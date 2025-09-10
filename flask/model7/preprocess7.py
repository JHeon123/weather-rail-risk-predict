import pandas as pd
import numpy as np
import torch

def preprocess_raw_input_v2(raw_input: dict,
                             feature_order: list,
                             label_encoders: dict,
                             imputer,
                             scaler) -> torch.Tensor:
    try:
        print("🔹 [1] 입력 dict → DataFrame 변환")
        df = pd.DataFrame([raw_input])
        print(df)

        print("🔹 [2] 범주형 인코딩 시작")
        for col, le in label_encoders.items():
            if col not in df.columns:
                raise KeyError(f"❌ 누락된 입력 컬럼: '{col}'")
            df[col] = df[col].astype(str)

            known_classes = le.classes_.tolist()
            df[col] = df[col].apply(lambda x: x if x in known_classes else '기타')
            if '기타' not in known_classes:
                le.classes_ = np.append(le.classes_, '기타')
            df[col] = le.transform(df[col])
        print("✅ 범주형 인코딩 완료")
        print(df)

        print("🔹 [3] 컬럼 순서 정렬")
        missing_cols = [col for col in feature_order if col not in df.columns]
        if missing_cols:
            raise ValueError(f"❌ 필요한 입력 컬럼 누락됨: {missing_cols}")
        df = df[feature_order]
        print(df)

        print("🔹 [4] 결측값 처리 (Imputer)")
        df_imputed = imputer.transform(df)
        print("✅ 결측값 처리 완료")
        print(df_imputed)

        print("🔹 [5] 스케일링 (Scaler)")
        df_scaled = scaler.transform(df_imputed)
        print("✅ 스케일링 완료")
        print(df_scaled)

        print("🔹 [6] 텐서 변환")
        input_tensor = torch.tensor(df_scaled, dtype=torch.float32)
        print("✅ 최종 텐서 shape:", input_tensor.shape)

        return input_tensor

    except Exception as e:
        import traceback
        traceback.print_exc()  # 🔥 전체 예외 스택 트레이스 출력
        raise
      
