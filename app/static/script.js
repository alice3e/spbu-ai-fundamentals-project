document.getElementById('prediction-form').addEventListener('submit', async function(event) {
    event.preventDefault();

    const formData = new FormData(this);
    const data = {};
    formData.forEach((value, key) => {
        // Convert empty strings for numbers to null so backend can handle them as NaN
        if (this.elements[key].type === 'number' && value === '') {
            data[key] = null;
        } else if (value === '' && this.elements[key].type !== 'text') { 
             // For non-text fields (like selects potentially, if not required), 
             // also send null if empty, unless it's an optional text field
            data[key] = null;
        }
         else {
            data[key] = value;
        }
    });

    // Ensure numerical fields are numbers if not null
    ['movie_year', 'budget', 'domestic', 'international'].forEach(key => {
        if (data[key] !== null && data[key] !== '') {
            data[key] = parseFloat(data[key]);
        }
    });
    
    // For genres/actors not filled, send as empty string or null, backend expects 'Unknown' or handles NaN
    ['genre_2', 'genre_3', 'genre_4', 'main_actor_4', 'director', 'writer', 'producer', 'composer', 'cinematographer', 'distributor'].forEach(key => {
        if (data[key] === '') {
            data[key] = null; // Let backend handle as NaN or 'Unknown'
        }
    });


    const resultP = document.getElementById('prediction-result');
    resultP.textContent = 'Predicting...';

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
        }

        const result = await response.json();
        resultP.textContent = `$${result.prediction.toLocaleString()}`;

    } catch (error) {
        console.error('Error:', error);
        resultP.textContent = `Error: ${error.message}`;
    }
});