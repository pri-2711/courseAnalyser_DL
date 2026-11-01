import { Fragment } from 'react'
import { Bar, BarChart, CartesianGrid, LabelList, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import './compare.css'

// Sentiment values are the stored SBERT reference predictions, the same source as the dashboard's rating chart.
const SENTIMENT = [['positive', 'Positive', '#6862d9'], ['negative', 'Negative', '#f18c91'], ['neutral', 'Neutral', '#aeb5c9']]
const SMALL_SAMPLE = 30

const shortTitle = (title = '') => (title.length > 28 ? `${title.slice(0, 27)}…` : title)
const titleCase = (text) => (text ? text.charAt(0).toUpperCase() + text.slice(1) : '—')
const count = (value) => (value == null ? '—' : Number(value).toLocaleString())
const pct = (share, digits = 1) => (share == null ? '—' : `${(share * 100).toFixed(digits)}%`)

// Analysed reviews are the rows the app holds for this course; published reviews come from the platform metadata.
const analysed = (c) => c.summary?.review_count ?? 0
const published = (c) => {
  if (c.metadata?.review_count == null || c.metadata.review_count === '') return null
  const value = Number(c.metadata.review_count)
  return Number.isFinite(value) ? value : null
}
const platformRating = (c) => {
  if (c.metadata?.rating == null || c.metadata.rating === '') return null
  const value = Number(c.metadata.rating)
  return Number.isFinite(value) ? value : null
}
// Pages outside the named platforms are stored as 'generic'; show the publisher instead of that label.
const platformName = (c) => (c.platform && c.platform !== 'generic' ? titleCase(c.platform) : c.metadata?.provider || 'Other website')
const netScore = (c) => (analysed(c) ? c.summary?.sentiment_value ?? null : null)

function sentimentShares(c) {
  const n = analysed(c), counts = c.summary?.sentiment_counts || {}
  return Object.fromEntries(SENTIMENT.map(([key]) => [key, { n: counts[key] || 0, share: n ? (counts[key] || 0) / n : null }]))
}

function dominantKey(c) {
  const shares = sentimentShares(c)
  return SENTIMENT.reduce((best, item) => (shares[item[0]].n > shares[best[0]].n ? item : best))[0]
}

function starSummary(c) {
  const rated = Object.values(c.summary?.rating_sentiment || {}).reduce((sum, row) => sum + Object.values(row).reduce((a, b) => a + b, 0), 0)
  return { rated, unrated: Math.max(analysed(c) - rated, 0), average: c.summary?.average_review_rating ?? null }
}

const labelOf = (key) => SENTIMENT.find((item) => item[0] === key)[1]

function buildInsights(list) {
  const items = []
  const withReviews = list.filter((c) => analysed(c) > 0)
  list.filter((c) => analysed(c) === 0).forEach((c) => items.push(`${c.title} has no analysed reviews, so no sentiment figures are shown for it.`))

  if (withReviews.length >= 2) {
    const dominant = new Set(withReviews.map(dominantKey))
    items.push(dominant.size === 1
      ? `Every compared course with analysed reviews has ${labelOf([...dominant][0]).toLowerCase()} as its largest predicted sentiment group.`
      : `The largest predicted sentiment group differs between courses: ${withReviews.map((c) => `${shortTitle(c.title)} (${labelOf(dominantKey(c)).toLowerCase()})`).join('; ')}.`)

    const positives = withReviews.map((c) => ({ c, share: sentimentShares(c).positive.share }))
    const high = positives.reduce((a, b) => (b.share > a.share ? b : a))
    const low = positives.reduce((a, b) => (b.share < a.share ? b : a))
    if (high.share !== low.share) {
      items.push(`Predicted positive share ranges from ${pct(low.share)} (${shortTitle(low.c.title)}, n = ${count(analysed(low.c))}) to ${pct(high.share)} (${shortTitle(high.c.title)}, n = ${count(analysed(high.c))}). These are model predictions on review text, not measured satisfaction.`)
    }

    const sizes = withReviews.map(analysed)
    if (Math.max(...sizes) >= 2 * Math.min(...sizes)) {
      items.push(`Review sample sizes differ a lot (${withReviews.map((c) => `${shortTitle(c.title)} n = ${count(analysed(c))}`).join(', ')}). Percentages from the smaller sample are less stable.`)
    }
  }

  const small = withReviews.filter((c) => analysed(c) < SMALL_SAMPLE)
  if (small.length) {
    items.push(`${small.map((c) => shortTitle(c.title)).join(', ')} ${small.length === 1 ? 'has' : 'have'} fewer than ${SMALL_SAMPLE} analysed reviews, so its percentages are indicative only.`)
  }

  const platforms = [...new Set(list.map(platformName))]
  if (platforms.length === 1) items.push(`All compared courses are on ${platforms[0]}.`)
  else if (platforms.length > 1) items.push(`The courses come from different platforms (${platforms.join(', ')}). Review populations and rating systems may differ, so compare these figures with care.`)

  const noRating = list.filter((c) => platformRating(c) == null)
  if (noRating.length) {
    items.push(`No platform rating is published for ${noRating.map((c) => shortTitle(c.title)).join(', ')}. These courses are left out of the rating chart rather than shown as zero.`)
  }
  const noPublishedCount = list.filter((c) => published(c) == null)
  if (noPublishedCount.length) {
    items.push(`No published review count is available for ${noPublishedCount.map((c) => shortTitle(c.title)).join(', ')}.`)
  }
  list.filter((c) => published(c) != null && analysed(c) < published(c)).forEach((c) => {
    items.push(`${shortTitle(c.title)}: ${count(analysed(c))} of ${count(published(c))} published reviews were analysed.`)
  })
  withReviews.forEach((c) => {
    const stars = starSummary(c)
    if (stars.unrated > 0) items.push(`${shortTitle(c.title)}: ${count(stars.unrated)} of ${count(analysed(c))} analysed reviews have no star rating, so they are not in rating comparisons.`)
  })
  return items
}

function OverviewCard({ c }) {
  const rating = platformRating(c), shares = sentimentShares(c), net = netScore(c)
  return <article className="cmp-card">
    <span className="platform-tag">{platformName(c)}</span>
    <h3>{c.title}</h3>
    <dl className="cmp-stats">
      <div><dt>Platform rating</dt><dd>{rating == null ? 'Not published' : `${rating.toFixed(1)} / 5`}</dd></div>
      <div><dt>Analysed reviews</dt><dd>{count(analysed(c))}</dd></div>
      <div><dt>Published reviews</dt><dd>{published(c) == null ? 'Not published' : count(published(c))}</dd></div>
      <div><dt>Net sentiment score</dt><dd>{net == null ? '—' : net.toFixed(2)}</dd></div>
    </dl>
    {analysed(c) > 0 ? <>
      <div className="cmp-split" role="img" aria-label="Predicted sentiment split">
        {SENTIMENT.map(([key, , color]) => shares[key].share > 0 && <span key={key} style={{ width: `${shares[key].share * 100}%`, background: color }}/>)}
      </div>
      <ul className="cmp-split-legend">
        {SENTIMENT.map(([key, label, color]) => <li key={key}><i style={{ background: color }}/>{label}<b>{pct(shares[key].share)}</b><small>n = {count(shares[key].n)}</small></li>)}
      </ul>
    </> : <p className="muted">No analysed reviews yet, so no sentiment split is available.</p>}
  </article>
}

function ComparisonTable({ list }) {
  const rows = [
    ['Platform', (c) => platformName(c)],
    ['Platform rating (published, out of 5)', (c) => (platformRating(c) == null ? 'Not published' : platformRating(c).toFixed(1))],
    ['Published review count', (c) => (published(c) == null ? 'Not published' : count(published(c)))],
    ['Analysed reviews', (c) => count(analysed(c))],
    ...SENTIMENT.map(([key, label]) => [`${label} (predicted)`, (c) => {
      const share = sentimentShares(c)[key]
      return share.share == null ? '—' : `${pct(share.share)} (n = ${count(share.n)})`
    }]),
    ['Net sentiment score (−1 to +1)', (c) => (netScore(c) == null ? '—' : netScore(c).toFixed(2))],
    ['Average star rating in analysed reviews', (c) => {
      const stars = starSummary(c)
      return stars.average == null ? '—' : `${stars.average.toFixed(2)} (n = ${count(stars.rated)} rated)`
    }],
    ['Analysed reviews with no star rating', (c) => count(starSummary(c).unrated)],
    ['Uncertain predictions', (c) => count(c.summary?.uncertain_review_count)],
  ]
  return <div className="cmp-table-wrap">
    <div className="cmp-table" style={{ gridTemplateColumns: `240px repeat(${list.length}, minmax(190px, 1fr))` }}>
      <div className="cmp-cell cmp-head cmp-label">Metric</div>
      {list.map((c) => <div className="cmp-cell cmp-head" key={c.id}>{shortTitle(c.title)}</div>)}
      {rows.map(([label, value]) => <Fragment key={label}>
        <div className="cmp-cell cmp-label">{label}</div>
        {list.map((c) => <div className="cmp-cell" key={c.id}>{value(c)}</div>)}
      </Fragment>)}
    </div>
  </div>
}

export default function CompareCourses({ courses, selected, setSelected }) {
  const list = courses.filter((c) => selected.includes(c.id))
  const withReviews = list.filter((c) => analysed(c) > 0)
  const shareData = withReviews.map((c) => {
    const shares = sentimentShares(c)
    return { name: `${shortTitle(c.title)} (n = ${count(analysed(c))})`, ...Object.fromEntries(SENTIMENT.map(([key, label]) => [label, shares[key].share * 100])) }
  })
  const volumeData = list.map((c) => ({ name: shortTitle(c.title), Analysed: analysed(c), Published: published(c) }))
  const ratingData = list.filter((c) => platformRating(c) != null).map((c) => ({ name: shortTitle(c.title), Rating: platformRating(c) }))
  const insights = list.length >= 2 ? buildInsights(list) : []

  return <>
    <div className="panel cmp-picker">
      <div className="panel-heading"><div><h3>Choose courses</h3><p>Select two or more saved analyses. Nothing here ranks courses or declares one better.</p></div></div>
      <div className="cmp-pick-grid">
        {courses.map((c) => <label className="cmp-pick" key={c.id}>
          <input type="checkbox" checked={selected.includes(c.id)} onChange={(e) => setSelected(e.target.checked ? [...selected, c.id] : selected.filter((x) => x !== c.id))}/>
          <span><b>{c.title}</b><small>{platformName(c)} · {count(analysed(c))} analysed reviews</small></span>
        </label>)}
      </div>
      {courses.length < 2 && <p className="muted">Add at least two courses to compare.</p>}
      {courses.length >= 2 && list.length < 2 && <p className="muted">Select at least two courses to see the comparison.</p>}
    </div>

    {list.length >= 2 && <>
      <section className="cmp-cards">{list.map((c) => <OverviewCard key={c.id} c={c}/>)}</section>

      <div className="panel cmp-panel">
        <div className="panel-heading"><div><h3>What the comparison shows</h3><p>Factual observations from the stored data only.</p></div></div>
        {insights.length ? <ul className="cmp-insights">{insights.map((text) => <li key={text}>{text}</li>)}</ul> : <p className="muted">No differences to report.</p>}
      </div>

      <div className="panel cmp-panel">
        <div className="panel-heading"><div><h3>Predicted sentiment share</h3><p>Share of each course's analysed reviews, by the stored SBERT reference prediction. The n next to each course is its analysed review count.</p></div></div>
        {shareData.length ? <div className="cmp-chart"><ResponsiveContainer width="100%" height={Math.max(180, 60 * shareData.length + 90)}>
          <BarChart data={shareData} layout="vertical" margin={{ top: 8, right: 24, left: 8, bottom: 8 }}>
            <CartesianGrid strokeDasharray="3 3" horizontal={false}/>
            <XAxis type="number" domain={[0, 100]} unit="%"/>
            <YAxis type="category" dataKey="name" width={200} tick={{ fontSize: 12 }}/>
            <Tooltip formatter={(value) => `${Number(value).toFixed(1)}%`}/>
            <Legend verticalAlign="top"/>
            {SENTIMENT.map(([, label, color]) => <Bar key={label} dataKey={label} stackId="share" fill={color}/>)}
          </BarChart>
        </ResponsiveContainer></div> : <div className="cmp-empty">None of the selected courses has analysed reviews yet.</div>}
      </div>

      <div className="cmp-grid-2">
        <div className="panel cmp-panel">
          <div className="panel-heading"><div><h3>Review volume</h3><p>Analysed reviews compared with the count published by the platform. Where no count is published, no bar is drawn.</p></div></div>
          <div className="cmp-chart"><ResponsiveContainer width="100%" height={280}>
            <BarChart data={volumeData} margin={{ top: 24, right: 8, left: 4, bottom: 8 }}>
              <CartesianGrid strokeDasharray="3 3" vertical={false}/>
              <XAxis dataKey="name" interval={0} tick={{ fontSize: 11 }}/>
              <YAxis allowDecimals={false}/>
              <Tooltip formatter={(value) => count(value)}/>
              <Legend verticalAlign="top"/>
              <Bar dataKey="Analysed" fill="#6862d9" radius={[5, 5, 0, 0]}><LabelList dataKey="Analysed" position="top" fontSize={11}/></Bar>
              <Bar dataKey="Published" fill="#c9c5f2" radius={[5, 5, 0, 0]}><LabelList dataKey="Published" position="top" fontSize={11}/></Bar>
            </BarChart>
          </ResponsiveContainer></div>
        </div>

        <div className="panel cmp-panel">
          <div className="panel-heading"><div><h3>Platform rating</h3><p>Rating as published by the platform, out of 5. This is not the model's sentiment.</p></div></div>
          {ratingData.length ? <div className="cmp-chart"><ResponsiveContainer width="100%" height={Math.max(180, 56 * ratingData.length + 80)}>
            <BarChart data={ratingData} layout="vertical" margin={{ top: 8, right: 40, left: 8, bottom: 8 }}>
              <CartesianGrid strokeDasharray="3 3" horizontal={false}/>
              <XAxis type="number" domain={[0, 5]} ticks={[0, 1, 2, 3, 4, 5]}/>
              <YAxis type="category" dataKey="name" width={160} tick={{ fontSize: 12 }}/>
              <Tooltip formatter={(value) => `${Number(value).toFixed(1)} / 5`}/>
              <Bar dataKey="Rating" fill="#6862d9" radius={[0, 5, 5, 0]}><LabelList dataKey="Rating" position="right" formatter={(value) => Number(value).toFixed(1)} fontSize={11}/></Bar>
            </BarChart>
          </ResponsiveContainer></div> : <div className="cmp-empty">No platform rating is published for the selected courses.</div>}
        </div>
      </div>

      <div className="panel cmp-panel cmp-explain">
        <h3>About the sentiment score</h3>
        <p><b>Net sentiment score</b> = (positive − negative) ÷ analysed reviews, ranging from −1 to +1. It summarises the predicted tone of the review text only. It does not measure course quality, instruction or value, and it is not a ranking.</p>
        <p><b>Platform ratings</b> are what the course page publishes. <b>Predicted sentiment</b> comes from the local model reading each review's text. The two are shown separately and are never combined.</p>
      </div>

      <div className="panel cmp-panel">
        <div className="panel-heading"><div><h3>Detailed comparison</h3><p>Every metric for the selected courses. Sample sizes are shown next to percentages.</p></div></div>
        <ComparisonTable list={list}/>
      </div>
    </>}
  </>
}
