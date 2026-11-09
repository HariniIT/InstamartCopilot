
import { useState } from 'react'
import './App.css'

function App() {
  const [complaint, setComplaint] = useState('')
  const [customerId, setCustomerId] = useState('CUST001')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const analyzeComplaint = async () => {
    if (complaint.trim().length < 5) {
      setError('Please enter at least 5 characters.')
      return
    }

    setLoading(true)
    setError('')
    setResult(null)

    try {
      const response = await fetch(
        'http://127.0.0.1:8000/api/complaints',
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            customer_id: customerId,
            complaint: complaint,
          }),
        }
      )

      const data = await response.json()

      if (!response.ok) {
        throw new Error(data.detail || 'Complaint analysis failed')
      }

      setResult(data)
    } catch (err) {
      setError(err.message || 'Backend connection failed.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="app">
      <aside className="sidebar">
        <h2>🛒 Instamart</h2>
        <p className="subtitle">AI COPILOT</p>

        <nav>
          <a className="active" href="#dashboard">▦ Dashboard</a>
          <a href="#customers">♙ Customers</a>
          <a href="#support">✉ Customer Support</a>
          <a href="#tickets">✦ AI Ticket Analysis</a>
          <a href="#refund">₹ Refund Assistant</a>
        </nav>

        <div className="sidebar-bottom">
          <span>● System Online</span>
          <small>Instamart Copilot v1.0</small>
        </div>
      </aside>

      <main className="main-content" id="dashboard">
        <header className="topbar">
          <div>
            <h1>Dashboard</h1>
            <p>Welcome to Instamart Customer Intelligence Platform</p>
          </div>
          <div className="profile">Admin</div>
        </header>

        <section className="welcome">
          <div>
            <h2>Customer Intelligence Dashboard</h2>
            <p>Monitor customers, orders and AI-powered support.</p>
          </div>
          <span>AI Powered</span>
        </section>

        <section className="stats">
          <div className="stat-card">
            <p>Total Customers</p>
            <h2>1,000</h2>
            <small>Registered customers</small>
          </div>

          <div className="stat-card">
            <p>Total Orders</p>
            <h2>3,000</h2>
            <small>Orders in dataset</small>
          </div>

          <div className="stat-card">
            <p>Total Products</p>
            <h2>100</h2>
            <small>Available products</small>
          </div>

          <div className="stat-card">
            <p>AI Support</p>
            <h2>Active</h2>
            <small>ML models integrated</small>
          </div>
        </section>

        <section className="content-grid">
          <div className="panel" id="support">
            <h3>Customer Support</h3>
            <p>Submit a customer complaint for AI analysis.</p>

            <label>Customer ID</label>
            <input
              type="text"
              value={customerId}
              onChange={(e) => setCustomerId(e.target.value)}
              placeholder="Enter customer ID"
            />

            <label>Customer Complaint</label>
            <textarea
              value={complaint}
              onChange={(e) => setComplaint(e.target.value)}
              placeholder="Example: I received a damaged product..."
            />

            <button onClick={analyzeComplaint} disabled={loading}>
              {loading ? 'Analyzing...' : 'Analyze Complaint'}
            </button>

            {error && (
              <p className="error-message">{error}</p>
            )}
          </div>

          <div className="panel" id="tickets">
            <h3>AI Ticket Analysis</h3>
            <p>AI-powered ticket classification and prioritization.</p>

            <div className="feature">✦ Category Prediction</div>
            <div className="feature">✦ Sentiment Analysis</div>
            <div className="feature">✦ Urgency Detection</div>
            <div className="feature">✦ Priority Prediction</div>
          </div>

          <div className="panel" id="refund">
            <h3>Refund Assistant</h3>
            <p>Get policy-based refund assistance.</p>

            <div className="feature">✓ Policy Retrieval</div>
            <div className="feature">✓ Refund Eligibility Support</div>
            <div className="feature">✓ Escalation Detection</div>
          </div>

          <div className="panel" id="customers">
            <h3>Customer Intelligence</h3>
            <p>Understand customer behaviour using ML.</p>

            <div className="feature">Customer Segmentation</div>
            <div className="feature">Churn Risk Prediction</div>
          </div>
        </section>

        {result && (
          <section className="panel result-panel">
            <h3>AI Complaint Analysis Result</h3>

            <div className="result-grid">
              <div className="feature">
                <strong>Customer ID</strong>
                <p>{result.customer_id}</p>
              </div>

              <div className="feature">
                <strong>Category</strong>
                <p>{result.category}</p>
              </div>

              <div className="feature">
                <strong>Sentiment</strong>
                <p>{result.sentiment}</p>
              </div>

              <div className="feature">
                <strong>Urgency</strong>
                <p>{result.urgency}</p>
              </div>

              <div className="feature">
                <strong>Priority</strong>
                <p>{result.priority}</p>
              </div>

              <div className="feature">
                <strong>Policy ID</strong>
                <p>{result.policy_id || 'No policy found'}</p>
              </div>
            </div>

            <h4>AI Response</h4>
            <p className="ai-response">{result.ai_response}</p>

            <h4>Human Escalation</h4>
            <p>
              {result.human_escalation
                ? '⚠ Human support required'
                : '✓ No human escalation required'}
            </p>
          </section>
        )}

        <footer>
          © 2026 Instamart Copilot | AI Customer Support Platform
        </footer>
      </main>
    </div>
  )
}

export default App
