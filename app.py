"""Run with: streamlit run app.py. Never accept uploaded pickle/joblib artifacts."""
import json
import os
from pathlib import Path
import joblib
import pandas as pd
import sklearn
import streamlit as st
from core import RAW, TYPES, score
from train import metrics

st.set_page_config(page_title='FraudWatch', page_icon='🛡️', layout='wide')
st.title('FraudWatch')
st.caption('Mobile-wallet transaction review • PaySim course project')
st.info('نموذج تعليمي على بيانات محاكاة. النتيجة درجة اشتباه وليست إثبات احتيال أو احتمالًا معايرًا.')
root = Path(os.environ.get('FRAUDWATCH_OUTPUT', 'outputs'))
path = root / 'model.joblib'
if not path.exists():
    st.warning('درّب النموذج أولًا. ضع ملف PaySim داخل data باسم paysim.csv ثم نفّذ:')
    st.code('python train.py --data data/paysim.csv')
    st.markdown('بعد اكتمال التدريب، أعد تحميل الصفحة. دليل التشغيل موجود في START_HERE_AR.md.')
    st.stop()

@st.cache_resource
def load_artifact(filename, modified):
    return joblib.load(filename)

try:
    artifact = load_artifact(str(path), path.stat().st_mtime_ns)
except Exception as exc:
    st.error('تعذر تحميل النموذج المحلي. أعد التدريب بنفس بيئة التشغيل. ' + str(exc))
    st.stop()
if artifact['sklearn_version'] != sklearn.__version__:
    st.error('إصدار scikit-learn مختلف عن إصدار التدريب. استخدم نفس الإصدار أو أعد التدريب.')
    st.stop()
report = artifact['report']
st.sidebar.header('Model information')
st.sidebar.write(artifact['model_name'])
st.sidebar.caption('العتبة الأصلية اختيرت على Validation فقط. تعديلها هنا استكشاف ولا يغيّر تقرير الاختبار.')
custom = st.sidebar.checkbox('Explore a different threshold', value=False)
threshold = st.sidebar.slider('Review threshold', 0., 1., min(1., float(artifact['threshold'])), .005) if custom else artifact['threshold']
st.sidebar.write(f'Active threshold: {threshold:.5f}')
st.sidebar.caption('No-review policy' if threshold > 1 else 'Scores at or above the threshold require review.')
tabs = st.tabs(['Single transaction', 'Batch CSV', 'Model evaluation', 'Threshold trade-off'])
with tabs[0]:
    with st.form('transaction'):
        c1, c2 = st.columns(2)
        kind = c1.selectbox('Transaction type', TYPES, index=4)
        amount = c1.number_input('Amount (dataset units)', min_value=0., value=1000.)
        step = c1.number_input('Simulation step (whole hour)', min_value=0, value=1)
        origin = c2.number_input('Origin balance BEFORE transaction', min_value=0., value=5000.)
        dest = c2.number_input('Destination balance BEFORE transaction', min_value=0., value=1000.)
        submitted = st.form_submit_button('Assess transaction', type='primary')
    if submitted:
        frame = pd.DataFrame([dict(step=step, type=kind, amount=amount, oldbalanceOrg=origin, oldbalanceDest=dest)])
        result = score(frame, artifact, threshold)
        st.metric('Model risk score', f'{result.risk_score.iloc[0]:.4f}')
        if result.review_required.iloc[0]:
            st.warning('تحتاج مراجعة بشرية. النموذج لا يثبت وقوع احتيال.')
        else:
            st.success('أقل من عتبة المراجعة الحالية. هذا لا يضمن سلامة العملية.')
with tabs[1]:
    st.write('Required columns: ' + ', '.join(RAW))
    st.caption('حد التطبيق: 100,000 صف و50 MB. درّب باستخدام train.py، وليس برفع ملف التدريب الكامل هنا.')
    template = pd.DataFrame([dict(step=1, type='TRANSFER', amount=1000, oldbalanceOrg=5000, oldbalanceDest=1000)])
    st.download_button('Download CSV format example (unlabeled)', template.to_csv(index=False), 'input_format.csv', 'text/csv')
    uploaded = st.file_uploader('Transactions CSV', type=['csv'])
    if uploaded and st.button('Score uploaded transactions', type='primary'):
        try:
            frame = pd.read_csv(uploaded, nrows=100001)
            if len(frame) > 100000:
                raise ValueError('الملف أكبر من 100,000 صف. قسّمه إلى ملفات أصغر.')
            result = score(frame, artifact, threshold)
            result.insert(0, 'csv_row', range(2, len(result)+2))
            result = result.sort_values('risk_score', ascending=False)
            a, b, c = st.columns(3)
            a.metric('Transactions', len(result))
            b.metric('Review required', int(result.review_required.sum()))
            c.metric('Review rate', f'{result.review_required.mean():.1%}')
            st.dataframe(result.head(1000), hide_index=True, use_container_width=True)
            st.download_button('Download all scored rows', result.to_csv(index=False), 'scored_transactions.csv', 'text/csv')
        except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as exc:
            st.error(str(exc))
with tabs[2]:
    st.subheader('Frozen held-out test result')
    st.caption('لا تستخدم هذه النتائج لإعادة اختيار النموذج. كل المقارنات الأساسية تستخدم عتبة 0.5، والبطاقات تعرض العتبة المختارة على Validation.')
    m = report['test_selected']
    cols = st.columns(4)
    for col, key in zip(cols, ['average_precision', 'precision', 'recall', 'f1']):
        col.metric(key.replace('_', ' ').title(), f'{m[key]:.3f}')
    st.dataframe(pd.read_csv(root / 'model_comparison.csv'), hide_index=True)
    c1, c2 = st.columns(2)
    c1.image(str(root / 'precision_recall.png'))
    c2.image(str(root / 'confusion_matrix.png'))
    with st.expander('Training data analysis'):
        st.image(str(root / 'eda.png'))
        st.image(str(root / 'correlations.png'))
        for insight in report['insights']:
            st.write('• ' + insight)
    with st.expander('Run details and limitations'):
        st.json(report)
        st.write('PaySim is simulated. Account identifiers are excluded, but repeat-account dependence may remain. New-customer generalization has not been established. Scores are not calibrated probabilities.')
with tabs[3]:
    st.subheader('Validation-only policy exploration')
    st.caption('التكاليف التالية افتراضات للمقارنة، وليست خسائر مالية مقاسة أو إيرادًا تم توفيره.')
    scores = pd.read_csv(root / 'validation_scores.csv')
    a, b = st.columns(2)
    fn_cost = a.number_input('Assumed cost per missed fraud', min_value=.01, value=float(report['cost_assumptions']['missed_fraud']))
    fp_cost = b.number_input('Assumed cost per false alarm', min_value=.01, value=float(report['cost_assumptions']['false_alarm']))
    vals = metrics(scores.isFraud, scores.risk_score, threshold)
    c1, c2, c3 = st.columns(3)
    c1.metric('Missed fraud', vals['fn'])
    c2.metric('False alarms', vals['fp'])
    c3.metric('Hypothetical cost units', f"{vals['fn'] * fn_cost + vals['fp'] * fp_cost:,.1f}")
    grid = pd.read_csv(root / 'validation_thresholds.csv')
    grid['assumed_cost'] = grid.fn * fn_cost + grid.fp * fp_cost
    st.line_chart(grid.sort_values('threshold').set_index('threshold')[['assumed_cost']])
