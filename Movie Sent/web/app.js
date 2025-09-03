const apiBase = 'http://localhost:5000';

async function predict() {
	const text = document.getElementById('review').value.trim();
	const model = document.getElementById('model').value;
	const result = document.getElementById('result');
	result.textContent = 'Analyzing...';
	if (!text) {
		result.textContent = 'Please enter a review.';
		return;
	}
	try {
		const res = await fetch(`${apiBase}/predict`, {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify({ text, model })
		});
		const data = await res.json();
		if (!res.ok) {
			result.textContent = `Error: ${data.error || 'Request failed'}`;
			return;
		}
		result.innerHTML = `
			<strong>Model:</strong> ${data.model}<br/>
			<strong>Prediction:</strong> ${data.label}<br/>
			<strong>Confidence:</strong> ${data.probability ? data.probability.toFixed(3) : 'n/a'}
		`;
	} catch (e) {
		result.textContent = 'Failed to contact API. Is it running?';
	}
}

document.getElementById('analyze').addEventListener('click', predict);
