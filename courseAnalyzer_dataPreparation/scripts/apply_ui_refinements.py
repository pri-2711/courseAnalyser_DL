"""One-time source edit for the UI refinement pass (run once; not part of the application).

Each replacement asserts that its target occurs exactly once, so nothing is patched in the wrong place.
"""
from pathlib import Path
import sys

MAIN = Path(__file__).resolve().parents[1] / "frontend" / "src" / "main.jsx"
source = MAIN.read_text(encoding="utf-8")


def replace_between(text, start, end, new):
    assert text.count(start) == 1, f"start marker found {text.count(start)} times: {start[:50]}"
    head, rest = text.split(start, 1)
    assert end in rest, f"end marker missing: {end[:50]}"
    body, tail = rest.split(end, 1)
    return head + new + end + tail


def replace_once(text, old, new):
    assert text.count(old) == 1, f"expected one match, found {text.count(old)}: {old[:70]}"
    return text.replace(old, new)


NEW_COMPONENTS = r'''const MODEL_TABS=[['mlp','MLP','Deep Learning · supervised PyTorch MLP'],['linear_svm','Linear SVM','Traditional Machine Learning'],['sbert_reference','SBERT reference','Transformer reference-similarity classifier']]
const clip=(text,limit)=>text&&text.length>limit?text.slice(0,limit).trimEnd()+'…':text

function ModelBreakdown({summary}){
  const breakdown=summary.model_breakdown, defaultModel=summary.default_model||'mlp'
  const [choice,setChoice]=useState(defaultModel)
  if(!breakdown)return <section className="panel model-breakdown"><div className="panel-heading"><div><h3>Sentiment by model</h3><p>Per-model statistics are generated when a course is analysed.</p></div></div></section>
  const selected=breakdown[choice]||breakdown[defaultModel]
  const tab=MODEL_TABS.find(([key])=>key===choice)
  const total=selected.total||0
  return <section className="panel model-breakdown">
    <div className="panel-heading"><div><h3>Sentiment by model</h3><p>Showing predictions from <b>{selected.model}</b>. Predictions from different models are never combined.</p></div><span className="soft-badge">{tab?tab[2]:''}</span></div>
    <div className="model-tabs" role="tablist" aria-label="Select model">{MODEL_TABS.map(([key,label])=><button key={key} role="tab" aria-selected={choice===key} className={choice===key?'selected':''} onClick={()=>setChoice(key)}>{label}{key===defaultModel&&<small> default</small>}</button>)}</div>
    {selected.available?<div className="model-stats">
      <p className="model-total">{total.toLocaleString()} predictions{selected.rows_without_prediction?` · ${selected.rows_without_prediction.toLocaleString()} reviews have no prediction from this model`:''}</p>
      {['positive','negative','neutral'].map(label=><div className={`model-bar ${label}`} key={label}><span>{label[0].toUpperCase()+label.slice(1)}</span><div className="bar-track"><i style={{width:`${((selected.percent[label]||0)*100).toFixed(1)}%`}}/></div><b>{(selected.counts[label]||0).toLocaleString()}</b><small>{selected.percent[label]!=null?`${(selected.percent[label]*100).toFixed(1)}%`:'–'}</small></div>)}
    </div>:<p className="muted">No predictions are stored for this model on this course yet.</p>}
  </section>
}

function RatingSentimentChart({summary}){
  const byStar=new Map([1,2,3,4,5].map(star=>[star,{rating:`${star}★`,Positive:0,Negative:0,Neutral:0}]))
  Object.entries(summary.rating_sentiment||{}).forEach(([key,values])=>{
    const rating=Number(key); if(!Number.isFinite(rating))return
    if(!byStar.has(rating))byStar.set(rating,{rating:`${rating}★`,Positive:0,Negative:0,Neutral:0})
    const row=byStar.get(rating); row.Positive+=values.positive||0; row.Negative+=values.negative||0; row.Neutral+=values.neutral||0
  })
  const data=[...byStar.entries()].sort((a,b)=>a[0]-b[0]).map(([,row])=>row)
  const rated=data.reduce((sum,row)=>sum+row.Positive+row.Negative+row.Neutral,0)
  const unrated=Math.max((summary.review_count||0)-rated,0)
  return <div className="panel chart-panel rating-sentiment">
    <div className="panel-heading"><div><h3>Rating vs sentiment</h3><p>Counts of reviews by observed star rating (x-axis), coloured by the SBERT reference prediction.</p></div></div>
    {rated<5?<div className="empty-chart">Too few rated reviews to compare ratings with sentiment ({rated} with a star rating). Upload reviews that include star ratings.</div>:<>
      <div className="rating-chart"><ResponsiveContainer width="100%" height={280}><BarChart data={data} margin={{top:24,right:8,left:4,bottom:28}}>
        <CartesianGrid strokeDasharray="3 3" vertical={false}/>
        <XAxis dataKey="rating" label={{value:'Observed star rating',position:'insideBottom',offset:-14}}/>
        <YAxis allowDecimals={false} label={{value:'Number of reviews',angle:-90,position:'insideLeft',offset:10}}/>
        <Tooltip formatter={(value,name)=>[Number(value).toLocaleString(),`${name} prediction`]}/>
        <Legend verticalAlign="top" height={28}/>
        <Bar dataKey="Positive" fill={COLORS[0]} radius={[5,5,0,0]}><LabelList dataKey="Positive" position="top" fontSize={11}/></Bar>
        <Bar dataKey="Negative" fill={COLORS[1]} radius={[5,5,0,0]}><LabelList dataKey="Negative" position="top" fontSize={11}/></Bar>
        <Bar dataKey="Neutral" fill={COLORS[2]} radius={[5,5,0,0]}><LabelList dataKey="Neutral" position="top" fontSize={11}/></Bar>
      </BarChart></ResponsiveContainer></div>
      <p className="chart-explainer">This chart shows how reviews with different observed star ratings are distributed across predicted sentiment categories. It helps compare the rating given by a reviewer with the sentiment predicted from their review text.</p>
      {unrated>0&&<p className="muted">{unrated.toLocaleString()} reviews have no star rating and are not placed on this chart.</p>}
    </>}
  </div>
}

function CourseOverview({active}){
  const md=active.metadata||active
  const fact=(label,value)=>value!=null&&value!==''?<div key={label}><span>{label}</span><b>{String(value)}</b></div>:null
  const facts=[fact('Provider',md.provider||active.platform),fact('Instructor / author',md.instructor),fact('Duration',md.duration),
    fact('Price',md.price!=null?(Number(md.price)===0?'Free':md.price):null),
    fact('Rating',md.rating!=null?`${Number(md.rating).toFixed(2)} / 5`:null),
    fact('Ratings on page',md.review_count!=null?Number(md.review_count).toLocaleString():null),
    fact('Level',md.level),fact('Language',md.language)].filter(Boolean)
  const unavailable=(md.fields_unavailable||[]).filter(field=>['provider','instructor','duration','price','rating','review_count','level','language'].includes(field)).map(field=>field.replaceAll('_',' '))
  return <section className="panel details-panel course-overview">
    <div><div className="eyebrow">COURSE OVERVIEW</div><h3>{md.title||active.title||'Untitled resource'}</h3>
      <p>{md.description||'A public description was not available for this resource.'}</p>
      {facts.length>0&&<div className="overview-facts">{facts}</div>}
      {unavailable.length>0&&<p className="muted">Not available from this site: {unavailable.join(', ')}.</p>}
      <p className="metadata-status">{md.review_status||'Reviews: local upload required.'}</p></div>
    {(active.url||md.url)&&<a href={active.url||md.url} target="_blank" rel="noreferrer">Open original resource <ArrowUpRight size={15}/></a>}
  </section>
}

function RecommendationsPage(){
  const [query,setQuery]=useState(''), [result,setResult]=useState(null), [busy,setBusy]=useState(false), [error,setError]=useState('')
  async function search(e){e.preventDefault();setBusy(true);setError('');try{setResult(await api('/recommendations',{method:'POST',body:JSON.stringify({query})}))}catch(err){setError(err.message);setResult(null)}finally{setBusy(false)}}
  return <><section className="welcome-row"><div><div className="eyebrow">CONTENT-BASED MATCHING</div><h1>Course recommendations</h1><p>Search your saved courses and resources by topic. Matching uses the local SBERT encoder; add more resources by URL to widen the catalogue.</p></div></section>
    <form className="panel recommend-form" onSubmit={search}><input value={query} onChange={e=>setQuery(e.target.value)} placeholder="e.g. python programming for beginners" maxLength={200} aria-label="Topic or skill"/><button className="button primary" disabled={busy||query.trim().length<3}>{busy?'Searching…':'Find resources'}</button></form>
    {error&&<div className="panel extraction-error" role="alert">{error}</div>}
    {result&&result.status!=='ok'&&<div className="panel"><p>{result.message}</p></div>}
    {result?.results?.length>0&&<div className="recommend-grid">{result.results.map(item=><article key={item.id} className="rec-card">
      <div className="rec-top"><span className="rec-platform">{item.platform}</span>{item.resource_type&&<span className="rec-type">{item.resource_type}</span>}<span className="rec-similarity" title="Cosine similarity between your query and this resource (0 to 1)">Similarity {Number(item.relevance).toFixed(2)}</span></div>
      <h3 className="rec-title">{item.title}</h3>
      {item.provider&&<p className="rec-provider">{item.provider}</p>}
      {item.description&&<p className="rec-desc">{clip(item.description,220)}</p>}
      <div className="rec-facts"><span>Rating <b>{item.rating!=null?`${Number(item.rating).toFixed(2)} / 5`:'not published'}</b></span>{item.review_count?<span>Ratings <b>{Number(item.review_count).toLocaleString()}</b></span>:null}<span>Analysed reviews <b>{item.reviews_analyzed||0}</b></span></div>
      {item.reviews_analyzed>0&&item.sentiment_counts&&<div className="rec-sentiment">{['positive','negative','neutral'].map(label=><span key={label} className={`chip ${label}`}>{label} {item.sentiment_counts[label]||0}</span>)}{item.sentiment_value!=null&&<small>sentiment value {Number(item.sentiment_value).toFixed(2)}</small>}</div>}
      <p className="rec-why"><b>Why recommended</b>{item.explanation}.</p>
      <div className="rec-footer"><span className="rec-score" title="Final ranking score">Score {Number(item.score).toFixed(3)}</span><a className="rec-link" href={item.url} target="_blank" rel="noopener noreferrer">Open original resource ↗</a></div>
    </article>)}</div>}
    {result?.skipped?.length>0&&<details className="panel"><summary>{result.skipped.length} saved resource(s) could not be matched</summary><ul>{result.skipped.map(s=><li key={s.id}>{s.title}: {s.reason}</li>)}</ul></details>}
    {result&&<details className="panel"><summary>How ranking works</summary><p>score = {result.weights.relevance} × similarity + {result.weights.rating} × (rating ÷ 5) + {result.weights.sentiment} × ((sentiment value + 1) ÷ 2). A signal a resource does not publish contributes 0. Resources below similarity {result.weights.minimum_relevance} are not shown.</p></details>}
  </>
}

'''


def main():
    text = source
    text = replace_between(text, "function ModelBreakdown({summary}){", "function ExtractionPanel(", NEW_COMPONENTS)
    text = replace_once(text, "import { Area, AreaChart, Bar, BarChart, CartesianGrid, Cell, Pie, PieChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'",
                        "import { Area, AreaChart, Bar, BarChart, CartesianGrid, Cell, Legend, LabelList, Pie, PieChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'")
    # Remove the old course overview block from the bottom of the active dashboard and place it first.
    start = '<section className="panel details-panel">'
    assert text.count(start) == 1, "details block marker"
    before, after = text.split(start, 1)
    block_end = after.index("</section>") + len("</section>")
    text = before + after[block_end:]
    text = replace_once(text, '{active?<><section className="metric-grid">', '{active?<><CourseOverview active={active}/><section className="metric-grid">')
    # Replace the old inline rating panel with the component.
    old_rating_start = '<div className="panel"><div className="panel-heading"><div><h3>Rating vs sentiment</h3>'
    assert text.count(old_rating_start) == 1, "rating panel marker"
    head, rest = text.split(old_rating_start, 1)
    end_marker = '</div></section><section className="panel upload-panel">'
    assert rest.count(end_marker) == 1, "rating panel end marker"
    text = head + '<RatingSentimentChart summary={summary}/></section><section className="panel upload-panel">' + rest.split(end_marker, 1)[1]
    # Model comparison table: scroll container and sticky first two columns.
    text = replace_once(text, '<table className="metric-table"><thead><tr><th>Model</th><th>Category</th>',
                        '<div className="table-scroll"><table className="metric-table"><thead><tr><th className="sticky-col sticky-1">Model</th><th className="sticky-col sticky-2">Category</th>')
    text = replace_once(text, "<td>{name.replaceAll('_',' ')}</td><td>{result.model_category||'–'}</td>",
                        "<td className=\"sticky-col sticky-1\">{name.replaceAll('_',' ')}</td><td className=\"sticky-col sticky-2\">{result.model_category||'–'}</td>")
    assert text.count("</table>") == 1, "table close count"
    text = text.replace("</table>", "</table></div>")
    MAIN.write_text(text, encoding="utf-8")
    print("applied UI refinements to", MAIN)
    return 0


if __name__ == "__main__":
    sys.exit(main())
