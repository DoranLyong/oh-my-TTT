# Linear Attention: $(QK^\top)V \to Q(K^\top V)$ 유도

## 1. 출발점: Softmax Attention (Eq.1)

i번째 출력 토큰:

$$O_i = \sum_j \frac{\exp(Q_i K_j^\top / \sqrt{d})}{\sum_l \exp(Q_i K_l^\top / \sqrt{d})} \cdot V_j$$

여기서 $Q_i$는 $\exp(\cdot)$ 안에 갇혀 있어 $\sum_j$ 밖으로 뺄 수 없다.

$$\exp(Q_i K_j^\top) \neq Q_i \cdot \exp(K_j^\top)$$

비선형 함수 안의 변수는 인수분해가 불가능하다.

---

## 2. Linear Kernel 적용 (Step 2)

Softmax kernel $\exp(Q_i K_j^\top)$를 linear kernel $Q_i K_j^\top$로 대체:

$$O_i = \frac{\sum_j (Q_i K_j^\top) \cdot V_j}{\sum_j (Q_i K_j^\top)}$$

이제 $Q_i K_j^\top$는 단순 곱셈이므로 **분배법칙**이 성립한다.

---

## 3. $Q_i$를 $\sum_j$ 밖으로 인수분해 (Step 3 → Eq.3)

$Q_i$는 summation index $j$에 의존하지 않는 상수이므로:

$$\sum_j (Q_i K_j^\top) V_j = Q_i \sum_j (K_j^\top V_j)$$

따라서:

$$O_i = \frac{Q_i \left(\sum_j K_j^\top V_j\right)}{Q_i \left(\sum_j K_j^\top\right)} \tag{Eq.3}$$

이것이 핵심 — 계산 순서가 $(QK^\top)V$에서 $Q(K^\top V)$로 바뀐다.

---

## 4. 왜 Softmax에서는 불가능한가?

| | $Q_i$의 위치 | 인수분해 |
|---|---|---|
| Softmax | $\exp(Q_i K_j^\top)$ 안에 갇힘 | 불가 |
| Linear | $Q_i K_j^\top$ 단순 곱셈 | 가능 |

$\exp(a \cdot b) \neq a \cdot \exp(b)$ 이므로,
softmax attention에서는 $Q_i$를 시그마 밖으로 뺄 수 없다.

---

## 5. 복잡도 변화

| 방식 | 계산 순서 | 중간 행렬 크기 | 복잡도 |
|---|---|---|---|
| Softmax | $(QK^\top)V$ | $N \times N$ | $O(N^2 d)$ |
| Linear | $Q(K^\top V)$ | $d \times d$ | $O(N d^2)$ |

$N \gg d$ (토큰 수 >> 차원)일 때 linear attention이 훨씬 효율적이다.

---

## 6. Eq.4로의 단순화

Eq.3에서 분모(scalar normalizer)를 무시하면:

$$O = Q(K^\top V) \;\triangleq\; Q \cdot W = \text{FC}(Q) \tag{Eq.4}$$

$K^\top V$를 하나의 weight matrix $W$로 보면, linear attention은 **query에 대한 fully-connected layer**와 동일하다.
