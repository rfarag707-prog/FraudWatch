"""Generate a measured report, 12-slide PPTX, and editable post after training."""
import html
from pathlib import Path
import pandas as pd

def make_deliverables(out, report):
    out = Path(out)
    m = report['test_selected']
    comparison = pd.read_csv(out / 'model_comparison.csv')
    cv = pd.read_csv(out / 'cv_comparison.csv')
    slides = [
        ('FraudWatch', ['Mobile-wallet fraud detection with classical machine learning.', 'PaySim simulation data. Course project.', 'Presenter: add your name and team.']),
        ('The decision', ['Flag transactions for human review before execution.', 'Missed fraud and false alarms have different consequences.', 'A model score supports review; it does not prove fraud.']),
        ('Dataset and sampling', [f"Read {report['source_rows']:,} rows; uniformly sampled {report['sample_rows']:,}.", 'Random priority sampling covers the whole file, independent of labels.', 'Source: PaySim on Kaggle. Financial values use dataset units.']),
        ('What the training data shows', report['insights'][:3]),
        ('Features available at decision time', ['Inputs: type, amount, oldbalanceOrg, oldbalanceDest, step.', 'Derived: hour, amount/balance ratio, origin shortfall.', 'Exclude future balances, isFlaggedFraud, account IDs and the target.']),
        ('Validation protocol', ['Split complete time steps: earliest 60% train, next 20% validation, last 20% test.', 'Three expanding-window folds on training steps tune hyperparameters.', 'Fit imputation, encoding and scaling inside each training fold.']),
        ('Model comparison', [f"Selected by training CV: {report['selected_model']}.", *[f"{r.model}: CV AP {r.cv_average_precision:.4f}" for r in cv.itertuples()], 'Baseline: constant training prevalence.']),
        ('Review threshold', [f"Validation-selected cutoff: {report['threshold']:.5f}.", f"Assumed costs: missed fraud {report['cost_assumptions']['missed_fraud']:g}, false alarm {report['cost_assumptions']['false_alarm']:g} units.", 'Minimize validation cost, then freeze model and threshold.', 'Costs are assumptions. Scores are not calibrated probabilities.']),
        ('Held-out test results', [f"Average precision: {m['average_precision']:.4f}; ROC-AUC: {m['roc_auc']:.4f}.", f"Precision: {m['precision']:.4f}; recall: {m['recall']:.4f}; F1: {m['f1']:.4f}.", f"Caught fraud: {m['tp']}; missed fraud: {m['fn']}; false alarms: {m['fp']}.", 'Average precision summarizes the PR curve; it is not trapezoidal PR-AUC.']),
        ('Errors and limitations', [f"Inspect {m['fn']} missed fraud cases and {m['fp']} false alarms.", 'Review error_examples.csv and errors_by_type.csv before presenting.', 'Simulation, sample uncertainty and repeated accounts limit generalization.', 'Do not claim deployment readiness or real monetary savings.']),
        ('Application demonstration', ['Score one transaction or upload a batch CSV.', 'Download flagged rows with row references and model scores.', 'Explore threshold costs on validation data only.', 'Add the public app URL and a screenshot after deployment.']),
        ('Recommendation and next steps', ['Use the model as a review prioritization prototype.', 'Validate on new data and define real review costs with stakeholders.', 'Check probability calibration and drift before operational use.', 'Add your final business interpretation after inspecting the errors.'])]
    css = 'body{margin:0;background:#e9eff6;font:20px Arial;color:#14283f}section{box-sizing:border-box;background:white;margin:24px auto;padding:48px;width:1100px;min-height:580px;border-top:12px solid #0f9e91}h1{font-size:38px}li{margin:24px 0;line-height:1.4}small{color:#52657a}@media print{section{margin:0;width:100%;height:95vh;break-after:page}}'
    body = ''.join(f'<section><small>FRAUDWATCH / {i:02d}</small><h1>{html.escape(title)}</h1><ul>' + ''.join(f'<li>{html.escape(s)}</li>' for s in points) + '</ul></section>' for i, (title, points) in enumerate(slides, 1))
    (out / 'presentation.html').write_text('<!doctype html><html><meta charset="utf-8"><title>FraudWatch</title><style>'+css+'</style><body>'+body+'</body></html>', encoding='utf-8')
    # Optional presentation dependency: training results remain valid if unavailable.
    try:
        from pptx import Presentation
        from pptx.util import Inches, Pt
        from pptx.dml.color import RGBColor
        prs = Presentation(); prs.slide_width = Inches(13.333); prs.slide_height = Inches(7.5)
        for i, (title, points) in enumerate(slides, 1):
            s = prs.slides.add_slide(prs.slide_layouts[6])
            s.background.fill.solid(); s.background.fill.fore_color.rgb = RGBColor.from_string('F5F8FC')
            def text(left, top, width, height, content, size, color, bold=False):
                tb = s.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
                tf = tb.text_frame; tf.word_wrap = True
                for j, line in enumerate(content):
                    p = tf.paragraphs[0] if j == 0 else tf.add_paragraph()
                    p.text = line; p.font.size = Pt(size); p.font.bold = bold
                    p.font.color.rgb = RGBColor.from_string(color); p.space_after = Pt(19)
            text(.7, .35, 12, .4, [f'FRAUDWATCH  /  {i:02d}'], 13, '0F9E91', True)
            text(.7, 1.05, 12, 1., [title], 34, '14283F', True)
            text(.8, 2.3, 11.8, 4.6, points, 23, '243E59')
        prs.save(out / 'presentation.pptx')
    except ImportError:
        print('Install python-pptx to also generate PowerPoint. HTML presentation is available.')
    report_html = '<!doctype html><meta charset="utf-8"><title>FraudWatch results</title><style>body{max-width:1000px;margin:40px auto;font:17px Arial;line-height:1.6;color:#14283f}table{border-collapse:collapse;font-size:13px}td,th{padding:8px;border:1px solid #ddd}img{max-width:100%}</style>'
    report_html += '<h1>FraudWatch: measured experiment</h1><p>PaySim simulation. Test results describe this sample only.</p>'
    report_html += '<h2>Model comparison at threshold 0.5</h2>' + comparison.to_html(index=False, float_format=lambda x:f'{x:.4f}')
    report_html += '<h2>Selected policy</h2><p>' + html.escape(str(m)) + '</p>'
    report_html += ''.join(f'<img src="{p}" alt="{p}">' for p in ['eda.png','correlations.png','precision_recall.png','confusion_matrix.png'])
    report_html += '<h2>Limitations</h2><p>Uncalibrated scores, simulated data, uniform-sample uncertainty and repeated-account dependence. No measured savings or production readiness claim.</p>'
    (out / 'report.html').write_text(report_html, encoding='utf-8')
    post = f'''Draft: review and personalize before posting. Add your actual contribution, team and links.

I built a course prototype for prioritizing suspicious mobile-wallet transactions using PaySim simulation data.
I compared a baseline and three classifiers with forward validation and preprocessing inside each fold.
On a held-out temporal test sample, the selected model achieved average precision {m['average_precision']:.4f}, precision {m['precision']:.4f}, and recall {m['recall']:.4f} at the validation-selected threshold.
My main lesson: [write what you learned from the errors].
These are simulation results, not evidence of real-world fraud prevention.
Demo: [add URL] | Repository: [add URL]
#MachineLearning #DataScience #NextGenInstitute
'''
    (out / 'linkedin_draft.txt').write_text(post, encoding='utf-8')
