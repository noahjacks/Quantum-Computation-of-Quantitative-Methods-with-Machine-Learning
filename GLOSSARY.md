# Glossary: every term in this project, in plain English

Read this first if words like "spread", "RFQ" or "qubit" are new to you.
Terms are grouped by topic, in the order you meet them.

## 1. Bonds

| Term | Meaning |
|---|---|
| **Bond** | A loan that can be traded. A company borrows money from investors and promises to pay interest and give the money back on a fixed date. Whoever owns the bond receives those payments. |
| **Corporate bond** | A bond issued by a company (as opposed to a government). This project only uses US corporate bonds. |
| **Issuer** | The company that borrowed the money. |
| **Face value (par)** | The amount repaid at the end, usually $1,000 per bond. Prices are quoted as a percentage of it: a price of 98 means $980 per $1,000. |
| **Coupon** | The yearly interest, as a percentage of face value. A 5% coupon pays $50 a year per $1,000, normally in two halves. |
| **Maturity** | The date the loan is repaid. "Time to maturity" is how many years are left. |
| **Accrued interest** | Interest earned since the last coupon payment. The buyer pays it to the seller on top of the price. |
| **Yield** | The annual return you would earn if you bought the bond today and held it to maturity. Price and yield move in opposite directions: if the price falls, the yield rises. |
| **Treasury** | A bond issued by the US government. Treated as risk-free, so its yield is the benchmark. |
| **Credit spread** | A corporate bond's yield minus the yield of a Treasury of the same maturity. It is the extra return investors demand for the risk that the company fails to pay. Wider spread = riskier company. |
| **Credit rating** | A grade (AAA, BBB, B ...) that agencies such as S&P and Moody's give to an issuer. **Investment grade (IG)** means BBB- or better; **high yield (HY)** means worse. Ratings are licensed and not in our free data. |
| **Duration** | How sensitive a bond's price is to a change in yield. Duration 7 means the price falls about 7% if yields rise by 1 percentage point. Long-dated bonds have high duration. |
| **Convexity** | A correction to duration for large yield moves. |
| **Total return** | What an investor actually earned over a period: price change plus interest received. |
| **Momentum / reversal** | Momentum: bonds that did well recently tend to keep doing well. Reversal: over very short or very long periods, winners tend to give some back. |
| **Beta** | How strongly a bond's return moves when some market factor moves (the whole bond market, the stock market, interest rates ...). Beta 2 to the bond market means the bond typically moves twice as much as the market. |
| **Volatility** | How much returns bounce around. |

## 2. How bonds are traded

| Term | Meaning |
|---|---|
| **Dealer** | A bank that stands ready to buy bonds from, and sell bonds to, investors. Bonds are not traded on an exchange; you trade with a dealer. |
| **Bid / ask (offer)** | The dealer's **bid** is the price at which it will buy from you; its **ask** is the price at which it will sell to you. The ask is higher. |
| **Bid-ask spread** | Ask minus bid. It is the dealer's fee for the service, and a measure of how hard a bond is to trade. |
| **Mark-up** | How far a dealer's quote is from its own estimate of the fair price. |
| **Liquidity** | How easy it is to trade without moving the price. A liquid bond trades often with a small bid-ask spread. Most corporate bonds are illiquid: many do not trade at all on a given day. |
| **RFQ (request for quote)** | How investors trade: they ask several dealers at once for a price on a specific bond and amount. Each dealer answers without seeing the others' answers; the investor picks the best one, or walks away. |
| **Fill** | A dealer's quote is "filled" if the investor accepts it. |
| **p(fill)** | The probability that a quote will be filled. This is what the whole project tries to predict. A dealer who knows p(fill) can quote more cleverly. |
| **TRACE** | The US system to which every corporate bond trade must be reported. It is the source of our price data. |
| **Customer buy / customer sell** | Whether the investor bought from the dealer (at the dealer's ask) or sold to the dealer (at the dealer's bid). |
| **Rule 144A bond** | A bond sold privately to large institutions only. |
| **Basis point (bp)** | One hundredth of a percent: 0.01%. |

## 3. Our data

| Term | Meaning |
|---|---|
| **Event** | One row of our data: one replayed quote request for one bond on one day, with 98 inputs describing the situation and the outcome (filled or not). |
| **Feature / input** | One number describing the situation before the quote, e.g. "the bond's spread yesterday". |
| **Replayed quote** | Our stand-in for real RFQs, which nobody publishes. Every real customer trade is treated as a request; a simple simulated dealer quotes; the quote counts as filled if it beats the price the customer really got. See `src/qon/target.py`. |
| **Reference price** | The price our dealer builds its quote around: the bond's last traded price, updated for how far the whole market has moved since. |
| **Kappa** | How many "usual half-spreads" our dealer adds to (or subtracts from) the reference price. Set once so that 37% of quotes fill. |
| **Calibration period** | The 40 trading days before the test window. Every one-off choice is made here, so that the test window is never used to make a choice. |
| **Look-ahead bias** | Accidentally using information that was not yet available at the time. It makes results look far better than they are. A test in this project proves we have none. |

## 4. Machine learning

| Term | Meaning |
|---|---|
| **Model** | A recipe, learned from past examples, that turns inputs into a prediction. |
| **Training / testing** | A model learns from the training events and is then scored on test events it has never seen. |
| **Backtest** | Replaying history: train on the past, test on the next day, move one day forward, repeat. |
| **Blinding** | Days deliberately left out between the training data and the test day. |
| **AUC** | The score. The probability that a randomly chosen filled quote got a higher p(fill) than a randomly chosen unfilled one. 0.5 = guessing, 1.0 = perfect. |
| **Brier score** | A second score that also checks the probabilities themselves are sensible, not just their order. Lower is better. |
| **Confidence interval (95%)** | The range a number would plausibly fall in if the experiment were repeated on other days. If the interval of a difference includes zero, the difference is not established. |
| **Bootstrap** | The way we compute those intervals: re-draw the test days at random many times and recompute. |
| **Hyper-parameter** | A setting of a model that is chosen, not learned (e.g. how deep a tree may grow). |
| **Cross-validation** | Choosing settings by repeatedly holding back part of the training data as a mini-test. |
| **Logistic regression (LR)** | The simplest model: a weighted sum of the inputs. It cannot combine inputs. |
| **Random forest (RF), gradient boosting (XGB)** | Models built from many decision trees. They can combine inputs. |
| **Neural network (NN)** | Layers of simple units; can also combine inputs. |
| **Kernel** | A similarity score between two events. |
| **SVM (support vector machine)** | A model that predicts from the outcomes of the most similar past events, with similarity defined by a kernel. |
| **RBF kernel** | The standard classical kernel: similarity falls off smoothly with distance. |
| **Bandwidth** | How quickly a kernel's similarity falls off. Too narrow and nothing looks similar to anything; too wide and everything looks the same. |

## 5. Quantum computing

| Term | Meaning |
|---|---|
| **Qubit** | The basic unit of a quantum computer. Unlike a bit, its state is a direction on a sphere, not just 0 or 1. |
| **Gate** | An operation on one or two qubits. Its strength is an angle. |
| **Circuit** | A sequence of gates. Ours has 109 qubits in a line and 216 two-qubit gates. |
| **Entanglement** | A link between qubits created by two-qubit gates: the state of one can no longer be described on its own. |
| **Measurement / shots** | Reading a qubit gives a random 0 or 1. To get a useful number the circuit is run many times ("shots", HSBC used 4096) and the results averaged. |
| **Expectation value** | That average. Each qubit is measured along three directions (X, Y, Z), giving three averages between -1 and +1. |
| **Feature map** | Using a circuit to turn one row of data into a new row of numbers: the data sets the gate angles, the measured averages are the output. |
| **PQFM (projected quantum feature map)** | HSBC's name for exactly that. |
| **Heisenberg gate** | The two-qubit gate used here. It partially swaps the states of two neighbouring qubits. |
| **Trotter step** | One round of gates over all neighbouring pairs. We use two rounds. |
| **Alpha** | A single number that scales all gate angles. |
| **Noise** | Everything that makes a real chip differ from the ideal: imperfect gates, misread qubits, a finite number of shots. |
| **Depolarizing noise** | The standard model of gate error: with small probability the qubits' state is scrambled. |
| **Readout error** | Reading a 0 as 1 or the reverse. |
| **Shot noise** | The randomness left in an average over a finite number of shots. |
| **Error mitigation** | Tricks that reduce the effect of noise. HSBC used two: **Pauli twirling** (turns awkward errors into simple random ones) and **TREX** (measures and corrects readout error). |
| **Drift** | The chip's error rates slowly change over minutes and hours, so results depend slightly on when a circuit ran. |
| **Heron** | IBM's chip family used by HSBC (133 or 156 qubits). **ibm_torino**, **ibm_fez** and **ibm_marrakesh** are individual chips. |
| **Transpile** | Translate a circuit into the specific gates and qubit layout of one chip. |
| **Simulator** | Ordinary software that computes what a quantum computer would output. Exact simulation is normally impossible for 109 qubits, but possible for this shallow circuit. |
| **Light cone** | The reason it is possible: in a shallow circuit each qubit is only affected by a few neighbours, so small pieces can be simulated separately. |
| **Matrix product state (MPS)** | A compact way to store the state of a line of weakly entangled qubits. Used by HSBC, and by us for the fidelity kernel. |
| **Fidelity quantum kernel (FQK)** | Similarity = overlap of two quantum states. |
| **Projected quantum kernel (PQK)** | Similarity = an RBF kernel on the measured averages of the two states. |
| **Exponential concentration** | The typical failure of quantum kernels with many qubits: every pair of different inputs has similarity near zero. |
| **Quantum advantage** | Doing something measurably better with a quantum computer than is possible without one. |
