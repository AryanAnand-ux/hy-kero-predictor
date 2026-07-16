import { Component } from 'react';

export default class ErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, errorInfo) {
    console.error("ErrorBoundary caught an error:", error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="error-fallback-container">
          <div className="error-fallback-card">
            <h2>⚠️ Render Error</h2>
            <p className="error-msg">{this.state.error?.toString()}</p>
            <p className="error-tip">Please try reloading the page or contact support if the issue persists.</p>
            <button className="btn" onClick={() => window.location.reload()}>
              Reload Application
            </button>
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}
