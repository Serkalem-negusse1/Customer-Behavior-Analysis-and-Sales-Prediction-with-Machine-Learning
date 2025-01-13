document.getElementById('predictBtn').addEventListener('click', async function () {
    const storeId = document.getElementById('store_id').value;
    const dayOfWeek = document.getElementById('day_of_week').value;
    const promo = document.getElementById('promo').value;

    if (!storeId || !dayOfWeek || !promo) {
        alert('Please fill in all fields');
        return;
    }

    const data = {
        store_id: parseInt(storeId),
        day_of_week: parseInt(dayOfWeek),
        promo: parseInt(promo)
    };

    try {
        const response = await fetch('http://127.0.0.1:5000/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        });

        if (response.ok) {
            const result = await response.json();
            document.getElementById('rfResult').textContent = result.prediction;
        } else {
            const error = await response.json();
            alert('Error: ' + error.error);
        }
    } catch (error) {
        alert('Error: ' + error.message);
    }
});