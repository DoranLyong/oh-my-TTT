`workflow_log/codeflow.md` 파일을 생성했습니다. 주요 내용 요약:

4단계 파이프라인으로 구성됩니다:

1. TTT (Test-Time Training) - 8장의 이미지로 512(= 8x8x8) 개 triplet 생성 후, prompt_embeddings [24, 1, 32, 1024]만 학습. 같은 anchor에 대한 3D 예측 일관성(L1 loss)을 자기지도 신호로 사용

2. Pairwise Inference - 56개 이미지 쌍(28쌍 x symmetrize)에 대해:

* Patch Embedding: [2, 3, 288, 512] → Conv2d → [2, 576, 1024]
* Deep Prompting: 32개 prompt 토큰 삽입 → [2, 608, 1024]
* 24-layer ViT Encoder (RoPE2D + Self-Attention)
* 8-layer 비대칭 Decoder (Self-Attn + Cross-Attn)
* Linear Head + PixelShuffle → [1, 288, 512, 3] (3D 포인트) + confidence

3. Global Alignment - MST 초기화 + 300 iter gradient descent로 모든 카메라 포즈/깊이/초점거리 최적화

4. Point Cloud 저장 - 마스크 적용 후 PLY 파일로 출력

핵심 수학적 아이디어는 exp depth 변환 ($\hat{n} \cdot (e^d - 1)$)으로 깊이를 양수 보장하면서 정밀도를 유지하고, TTT loss는 anchor 이미지의 3D 좌표가 reference에 무관하게 일관되어야 한다는 조건을 활용합니다.

---

workflow_log/TTT_mechanism.md 파일을 생성했습니다. 주요 내용:

1. TTT 핵심 개념 — 사전학습 모델을 테스트 시점에 장면 특화로 적응시키는 기법. 전체 파라미터(571M) 중 prompt만(0.79M, 0.14%) 학습

2. Cross-pair Inconsistency 문제 — 같은 ref 이미지라도 다른 source와 쌍지으면 다른 3D 좌표 예측 → 이것이 TTT가 필요한 근본 이유

3. Triplet Consistency Loss — $\mathcal{L} = |P_1^{ref} - P_2^{ref}|$ (L1), ref의 3D 예측이 source에 무관하게 일관되도록 강제. 코드 train_criterion과 수학 수식 대응

4. Visual Prompt Tuning — Deep prompting([24,1,32,1024])이 매 encoder 레이어에서 prompt 토큰을 교체하는 방식. Self-Attention에서 patch→prompt attend를 통해 정보 주입

5. 전체 텐서 Shape 추적 — [2,3,288,512] → PatchEmbed → [2,608,1024] (prompt 포함) → 24-layer encoder → [2,576,1024] → decoder → head → [1,288,512,3] (pts3d)

6. Gradient Flow — loss에서 prompt_embeddings까지 gradient가 attention weight를 통해 역전파되는 경로 상세 분석. 2번의 forward pass에서 gradient가 합산되는 구조 설명