# 🏄 SurfCast Sri Lanka: සරල සිංහල මාර්ගෝපදේශය (Guide)

> මේ guide එකෙන් project එකේ හැම කොටසක්ම සරලව පැහැදිලි කරනවා, run කරන විදිහ කියනවා, 3-minute video script එකත් දෙනවා.
> **[ ] කොටු තියෙන තැන්වලට** notebook එක run කළාට පස්සේ `results/metrics.csv` එකේ හෝ dashboard එකේ තියෙන **ඇත්ත අගයන්** දාන්න. කවදාවත් අගයන් හදලා (invent කරලා) දාන්න එපා.

---

## 1. Project එක කෙටියෙන්

**ප්‍රශ්නය:** "හෙට **Arugam Bay** හෝ **Hikkaduwa** වල surf කරන්න හොඳ දවසක්ද?"

* ලංකාවේ surf tourism එකෙන් surf schools, guesthouses, tuk-tuk drivers, board rental shops වලට විශාල ආදායමක් ලැබෙනවා.
* වෙරළ දෙකේ **season එක ප්‍රතිවිරුද්ධයි**, මොකද monsoon සුළඟ දිවයිනේ පැත්ත මාරු කරනවා:
  * **නිරිත දිග monsoon (මැයි–සැප්තැම්බර්):** Arugam Bay වල සුළඟ ගොඩබිමේ සිට මුහුදට (**offshore**) හමනවා → රළ පිරිසිදුයි → **Arugam season**.
  * **ඊසාන දිග monsoon (නොවැම්බර්–මාර්තු):** නිරිත දිග වෙරළේ සුළඟ offshore → **Hikkaduwa season**.
* මේක **binary classification** ප්‍රශ්නයක්: හෙට = "හොඳ surf දවසක්" (1) හෝ "නැහැ" (0).
* අපි භාවිතා කරන්නේ **අද දවස අවසානයේ දන්න තොරතුරු පමණයි** (හෙට ගැන කිසිම දෙයක් model එකට දෙන්නේ නැහැ).

## 2. Data (ශ්‍රී ලාංකික, නොමිලේ, login නැහැ)

| මූලාශ්‍රය | තියෙන දේ |
|---|---|
| Open-Meteo **Marine API** | රළ / swell උස, period (රළ දෙකක් අතර කාලය), දිශාව |
| Open-Meteo **Historical Weather API** (ERA5) | සුළඟේ වේගය, gusts, දිශාව, වැසි, ආර්ද්‍රතාව, පීඩනය, වලාකුළු |

* ස්ථාන 2ක්, දිනපතා, **2010–2025** (අවුරුදු 16).
* `download_data.py` run කළාම data `data/raw/` folder එකට download වෙනවා (retry + caching එක්ක).

### "හොඳ surf දවසක්" (label එක) තීරණය කරන නීති
1. Swell උස **0.8 m – 2.5 m** අතර (පොඩි වැඩි නැහැ, භයානක තරම් ලොකුත් නැහැ)
2. Swell period **≥ 9 s** (දිගු period = ශක්තිමත්, පිරිසිදු රළ)
3. Swell එක එන්නේ ඒ වෙරළට ළඟා විය හැකි දිශාවෙන් (**swell window**)
4. සුළඟ **≤ 20 km/h** නම් ඕනම දිශාවක් OK; **offshore** නම් **≤ 35 km/h** දක්වා OK

## 3. Files

| File | කාර්යය |
|---|---|
| `download_data.py` | Open-Meteo වලින් data download කරනවා |
| `features.py` | Feature engineering. Notebook එකයි dashboard එකයි **දෙකම මේක පාවිච්චි කරනවා** (එකම විදිහට features හැදෙන්න) |
| `SurfCast_SriLanka.ipynb` | සම්පූර්ණ විශ්ලේෂණය: EDA → features → models → interpretation → models save |
| `app.py` | Streamlit dashboard (front end) |
| `models/*.pkl` | Save කළ models (backend), joblib වලින් |
| `results/*.csv` | Dashboard එක කියවන metrics / predictions |
| `requirements.txt` | අවශ්‍ය Python packages |

## 4. Run කරන විදිහ (Windows PowerShell)

```powershell
cd SurfCast_SL
python -m venv .venv
.\.venv\Scripts\Activate.ps1
# "running scripts is disabled" error එකක් ආවොත්:  Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
pip install -r requirements.txt
python download_data.py
$env:PYTHONIOENCODING="utf-8"
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 SurfCast_SriLanka.ipynb
streamlit run app.py
```
* Python **3.12 හෝ 3.13** නිර්දේශිතයි.
* `download_data.py` එකට විනාඩි 4–5ක් යනවා (free API limit එක නිසා calls අතර විනාඩියක් ඉන්නවා).
* Notebook එක run වෙන්න විනාඩි 10–20ක් විතර යනවා (laptop එක අනුව).
* Dashboard එක browser එකේ `http://localhost:8501` වලින් open වෙනවා.

---

## 5. Lecture සංකල්ප සරලව (viva එකට වැදගත්)

### Bias, Variance, Irreducible error
* **Test error = Bias² + Variance + Irreducible error**
* **Bias** = model එක ඕනවට වඩා සරලයි, රටාව අල්ලගන්න බැහැ (underfitting).
* **Variance** = model එක training data කටපාඩම් කරනවා, අලුත් data වලට වැරදියි (overfitting).
* **Irreducible error** = data වල ස්වාභාවික noise එක (උදා: grid model data, හදිසි local සුළං). කිසිම model එකකට අයින් කරන්න බැහැ.
* Notebook එකේ chart එක: **තනි Decision Tree** එකේ train accuracy එක ඉහළයි, test accuracy එක අඩුයි → **ලොකු gap = high variance**. Ensembles මේ gap එක අඩු කරනවා.

### Bagging (Bootstrap Aggregating)
* Training data වලින් **අහඹු ලෙස (replacement සමඟ) sample** කරලා trees ගොඩක් **එකවර (parallel)** හදනවා, ඊට පස්සේ ඒවගේ ඡන්දය / සාමාන්‍යය ගන්නවා.
* ඉලක්කය: **variance අඩු කිරීම**.
* උදාහරණ: **Bagged Trees**, **Random Forest**.

### Random Forest සහ *m* (`max_features`)
* RF = bagging + **හැම split එකකදීම features *m* ප්‍රමාණයක් පමණක් අහඹු ලෙස බලනවා**.
* එතකොට trees එකිනෙකට වෙනස් වෙනවා (de-correlated) → සාමාන්‍යය ගත්තම වඩා හොඳයි.
* **m = p** (සියලු features) නම් → ඒක **සාමාන්‍ය bagging** එකම තමයි.
* Classification වලට සාමාන්‍ය තේරීම **m = √p**.

### Out-of-Bag (OOB) error
* හැම bootstrap sample එකකටම rows වලින් **~37%ක් ඇතුළත් වෙන්නේ නැහැ** (out-of-bag).
* ඒ rows ඒවා නොදැක්ක trees වලින් predict කරලා **නොමිලේ validation error එකක්** ගන්න පුළුවන්.
* Notebook chart: trees ගණන වැඩි වෙද්දී OOB error එක ස්ථාවර වෙනවා, ඒක test error එකට ළඟයි.

### Boosting
* Models **එකින් එක (sequential)** හදනවා. **අලුත් model එක කලින් ඒවා කරපු වැරදි නිවැරදි කරනවා.**
* ඉලක්කය: **bias අඩු කිරීම**.
* **AdaBoost:** වැරදියට classify වුණු දවස්වලට වැඩි බරක් දෙනවා.
* **Gradient Boosting:** අලුත් tree එක කලින් model එකේ errors (gradients) වලට fit කරනවා.
* **XGBoost:** regularisation එක්ක වේගවත් gradient boosting.
* **LightGBM:** histogram + leaf-wise වර්ධනය, ඉතා වේගවත්.

### Stacking සහ Soft Voting
* **Stacking:** වෙනස් models කිහිපයක් (RF, XGBoost, LightGBM, Logistic Regression) → ඒවගේ predictions **meta-model** එකකට (Logistic Regression) දෙනවා → meta-model එක ඒවා එකතු කරන හොඳම විදිහ ඉගෙන ගන්නවා.
* **Soft Voting:** models වල probabilities වල සාමාන්‍යය.
* **වැදගත් පාඩම:** කාලානුක්‍රමික (time-ordered) data වල **shuffled K-fold** පාවිච්චි කළොත් අල්ලපු දවස් training/validation දෙකේම වැටෙනවා → meta-model එක එක base model එකකට ඕනවට වඩා විශ්වාස කරන්න පුළුවන්. ඒ නිසා අපි **blocked (shuffle නැති) K-fold** පාවිච්චි කළා, shuffled එකත් එක්ක සංසන්දනයත් කළා.

### Bagging vs Boosting vs Stacking

| | Bagging | Boosting | Stacking |
|---|---|---|---|
| ක්‍රමය | Parallel, bootstrap samples | Sequential, වැරදි නිවැරදි කරමින් | වෙනස් models + meta-model |
| Base models | එකම වර්ගය (ගැඹුරු trees) | එකම වර්ගය (නොගැඹුරු trees) | වෙනස් වර්ග |
| ඉලක්කය | Variance ↓ | Bias ↓ | ශක්තීන් එකතු කිරීම |
| එකතු කරන විදිහ | ඡන්දය / සාමාන්‍යය | බර සහිත එකතුව | Meta-model ඉගෙන ගනී |

### Metrics: accuracy ප්‍රමාණවත් නැත්තේ ඇයි?
* **Accuracy:** හරි predictions ප්‍රතිශතය. නමුත් "හොඳ දවස්" අඩු නම්, හැම දාම "නැහැ" කිව්වත් accuracy ඉහළ පේනවා.
* **Precision:** "හොඳයි" කියපු දවස්වලින් ඇත්තටම හොඳ වුණේ කීයද? (false alarm අඩුද?)
* **Recall:** ඇත්ත හොඳ දවස්වලින් අපි අල්ලගත්තේ කීයද? (surf school එකක් හොඳ දවසක lesson cancel කළොත් මුදල් අහිමියි)
* **F1:** Precision සහ Recall දෙකේ සමබරතාව → **අපේ ප්‍රධාන metric එක**.
* **ROC-AUC:** ඕනම threshold එකකදී model එක දවස් හරියට පෙළගස්වන (rank) හැකියාව.

### Decision threshold
* Model එක probability එකක් දෙනවා (උදා: 0.63). Threshold එකට (default 0.5) වඩා වැඩි නම් "හොඳයි".
* Threshold **අඩු කළොත්** → recall ↑, precision ↓. **වැඩි කළොත්** → precision ↑, recall ↓.
* Dashboard එකේ **live slider** එකෙන් මේක පෙන්නන්න පුළුවන්.

### Hyperparameter tuning
* **RandomizedSearchCV** + **TimeSeriesSplit** (validation එක හැමවිටම training වලට *පස්සේ* එන කාලයෙන්) + **scoring = F1**.
* XGBoost සහ Random Forest tune කරලා default එක්ක සංසන්දනය කළා.

### Time split සහ leakage
* **Random split එකක් කළොත්** මාර්තු 14 train එකේ, මාර්තු 15 test එකේ වැටෙනවා. ඒ දෙක ගොඩක් සමානයි → test score එක බොරුවට ඉහළ යනවා (**leakage**).
* ඒ නිසා: **පරණ අවුරුදු වලින් train කරලා, අවසාන අවුරුදු 3න් test** කළා.
* **sin/cos encoding:** මාසය, දිශාව වැනි චක්‍රීය අගයන් (359° සහ 1° ළඟයි) sin සහ cos වලින් encode කළා.

### Hybrid models
* **RF + SARIMA:** SARIMA එක මාසික swell එකේ **අවුරුදු චක්‍රය (monsoon seasonality)** අල්ලනවා. RF එක SARIMA කරන **වැරදි (residuals)** ඉගෙන ගන්නවා. Hybrid = SARIMA forecast + RF නිවැරදි කිරීම.
* **RF + MLP (ANN):** හෙට swell උස (meters) predict කරන regression එකක්. RF එක threshold-type නීති වලට හොඳයි, MLP එක smooth functions වලට හොඳයි. ඒ දෙකේ **බර සහිත සාමාන්‍යය** ගත්තා (බර තෝරුවේ training data වල අවසාන අවුරුද්දෙන්, test එක ස්පර්ශ නොකර).
* **Persistence baseline:** "හෙට = අද" කියන සරල අනාවැකිය. ඕනම ML model එකක් මේකට වඩා හොඳ විය යුතුයි.

---

## 6. Dashboard tabs

| Tab | පෙන්නන දේ |
|---|---|
| 🌊 Overview | ප්‍රධාන අගයන්, සිතියම, monsoon "see-saw" chart එක |
| 🔎 Data explorer | ඕනම variable එකක් කාලය අනුව, මාසික boxplot, CSV download |
| 📊 Model comparison | Metrics table, **live threshold slider**, confusion matrix, ROC, bias-variance |
| 🌲 Inside Random Forest | OOB error, *m* curve, RF vs XGBoost importance, tuning |
| 🎯 Predict | Test කාලයේ ඇත්ත දවසක් තෝරලා, what-if sliders, හැම model එකකම probability, **ඇත්ත ප්‍රතිඵලය** |
| 🔗 Hybrid models | RF + SARIMA සහ RF + MLP ප්‍රතිඵල |
| ℹ️ About | ක්‍රම, data, AI declaration |

## 7. ප්‍රතිඵල (notebook run කළාට පස්සේ පුරවන්න)

| කරුණ | අගය |
|---|---|
| හොඳම model එක (F1 අනුව) | [ ] |
| ඒකේ F1 / ROC-AUC | [ ] / [ ] |
| Persistence baseline F1 | [ ] |
| Decision Tree train vs test accuracy (gap) | [ ] vs [ ] |
| Random Forest train vs test accuracy | [ ] vs [ ] |
| වැදගත්ම features 3 (RF) | [ ] |
| Stacking blocked vs shuffled F1 | [ ] vs [ ] |
| RF+SARIMA hybrid RMSE vs SARIMA RMSE | [ ] vs [ ] |
| RF+MLP hybrid RMSE vs RF RMSE | [ ] vs [ ] |

---

## 8. 🎬 3-minute video script (English voice-over)

> Speak slowly and clearly. The words in *[brackets]* are the screen actions. Replace **[ ]** with the real numbers from your dashboard.

**0:00–0:20 · Intro and why it matters** · *[Overview tab: hero banner + map]*
"Every year thousands of surfers come to Sri Lanka, and local surf schools and guesthouses depend on them. But a surf day can be perfect or a total waste. This is SurfCast Sri Lanka, an ensemble-learning dashboard that predicts whether tomorrow will be a good surf day at Arugam Bay and Hikkaduwa."

**0:20–0:45 · Data and EDA** · *[Overview: monthly see-saw chart → Data explorer: swell height over time]*
"I used sixteen years of free Open-Meteo wave and weather data for both spots. Look at this see-saw. Arugam Bay peaks in the south-west monsoon, and Hikkaduwa peaks in the north-east monsoon, because the offshore wind switches sides of the island. A good surf day means the right swell size, a long wave period, the right direction and light or offshore wind."

**0:45–1:30 · Model comparison, bias-variance and the threshold slider** · *[Model comparison tab: table → train vs test chart → move the threshold slider]*
"I trained a single decision tree, bagging models, boosting models, stacking and soft voting, and tested them on the most recent three years only, so the future never leaks into training. The single tree overfits: [ ] accuracy on training but only [ ] on test. Bagging and boosting close that gap. The best model, [ ], reaches an F1 of [ ], compared with [ ] for the simple 'tomorrow equals today' baseline. Because good days are the minority, I focus on F1 and recall, not just accuracy. With this slider, a surf school can lower the threshold to catch more good days, or raise it to avoid false alarms."

**1:30–1:55 · Inside Random Forest** · *[Inside Random Forest tab: OOB curve → m curve → importance]*
"Inside the Random Forest, the out-of-bag error settles after about [ ] trees and tracks the real test error, so it works as free validation. Using only a few features per split beats plain bagging, where m equals p. Both Random Forest and XGBoost agree that [ ] and [ ] matter most."

**1:55–2:25 · Live prediction demo** · *[Predict tab: pick a day → show probabilities → move the wind slider]*
"Let's pick a real day from the test period at Arugam Bay. All the models give their probability for tomorrow, and here's what actually happened: [ ]. Now watch. If I turn the wind onshore and make it stronger, every model drops its prediction. The models have learned real surf physics."

**2:25–2:50 · Hybrids and honest lessons** · *[Hybrid models tab]*
"I also built two hybrids. RF plus SARIMA forecasts monthly swell, with the forest correcting SARIMA's errors, and RF plus a neural network predicts tomorrow's wave height. [Say honestly whether each hybrid beat its parts, with the numbers.] One more lesson: stacking with shuffled cross-validation on time data can mislead the meta-model, so I used blocked folds and reported the results honestly."

**2:50–3:00 · Closing** · *[Overview tab]*
"Simple ensemble models give strong results at low cost, so you don't always need deep learning. Thanks for watching SurfCast Sri Lanka."

---

## 9. Lecturer අහන්න පුළුවන් ප්‍රශ්න සහ කෙටි පිළිතුරු

1. **Random split එකක් නොකළේ ඇයි?** අල්ලපු දවස් සමානයි. Random split එකක් කළොත් leakage වෙලා test score එක බොරුවට ඉහළ යනවා.
2. **Bagging variance අඩු කරන්නේ කොහොමද?** ස්වාධීන (bootstrap) trees ගොඩක සාමාන්‍යය ගත්තම, එක් එක් tree එකේ අහඹු වැරදි එකිනෙක අහෝසි වෙනවා.
3. **RF bagging වලට වඩා හොඳ ඇයි?** හැම split එකකම features *m* ප්‍රමාණයක් පමණක් බලන නිසා trees අතර correlation අඩුයි, ඒ නිසා සාමාන්‍යය වඩා ශක්තිමත්.
4. **Boosting overfit වෙන්න පුළුවන්ද?** ඔව්. Rounds ගොඩක් හෝ learning rate එක වැඩි නම් වෙන්න පුළුවන්. ඒකයි අඩු learning rate සහ tuning පාවිච්චි කළේ.
5. **F1 තේරුවේ ඇයි?** Classes අසමතුලිතයි, ඒ නිසා accuracy නොමඟ යවනවා. F1 එකෙන් missed good days (recall) සහ false alarms (precision) දෙකම සලකනවා.
6. **OOB error එක මොකක්ද?** Bootstrap එකට ඇතුළත් නොවුණු ~37% rows වලින් ගන්නා නොමිලේ validation error එකක්.
7. **sin/cos ඇයි?** 359° සහ 1° ළඟයි. සාමාන්‍ය අංකයක් ලෙස දුන්නොත් model එක ඒවා ඈතයි කියලා හිතනවා.
8. **සීමාවන් මොනවාද?** Grid model data (buoy නොවේ), tide සහ sandbank සලකා නැහැ, label එක සරල නීති මත.

---

## 10. ✅ ඔබ තවමත් කළ යුතු දේ

- [ ] `pip install -r requirements.txt` → `python download_data.py` → notebook එක run කරන්න (errors 0ක් තියෙන්න ඕනේ)
- [ ] Notebook එකේ **"Interpretation"** කොටස් සහ **Conclusions** ඇත්ත අගයන් අනුව ලියන්න (නැත්නම් `results/` folder එක Claude ට දීලා ලියවගන්න)
- [ ] Notebook එකේ, dashboard එකේ (`app.py` sidebar + About tab) සහ README එකේ **AI declaration** එකට **ඔබේ නම සහ index number** දාන්න
- [ ] මේ guide එකේ සහ README එකේ **[ ]** කොටු ඇත්ත අගයන්ගෙන් පුරවන්න
- [ ] Notebook එක හොඳින් කියවලා තේරුම් ගන්න (viva එකට)
- [ ] `streamlit run app.py` කරලා හැම tab එකක්ම බලන්න
- [ ] Video එක record කරන්න (උපරිම විනාඩි 3, voice-over). OBS Studio හෝ Windows **Win + Alt + R** පාවිච්චි කරන්න පුළුවන්
- [ ] (Bonus) GitHub වලට push කරලා Streamlit Community Cloud එකේ host කරන්න
- [ ] **සඳුදා, නොවැම්බර් 23 මධ්‍යම රාත්‍රියට පෙර** `.ipynb` + video + links submit කරන්න
