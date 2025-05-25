// static/script.js
document.getElementById('prediction-form').addEventListener('submit', async function(event) {
    event.preventDefault();

    const formData = new FormData(this);
    const data = {};
    let hasEmptyRequiredSelect = false;

    formData.forEach((value, key) => {
        const element = this.elements[key];
        
        // Обработка Select2, которые могут возвращать null, если не выбраны и allowClear=true
        if (element.tagName === 'SELECT' && $(element).hasClass('select2-enable')) {
            // $(element).val() вернет null если ничего не выбрано и allowClear, или значение
            // Если множественный выбор, $(element).val() вернет массив. У нас одиночный.
            data[key] = $(element).val() === null || $(element).val() === "" ? null : $(element).val();
        } else if (element.type === 'number' && value === '') {
            data[key] = null; // Пустые числа как null
        } else if (value === '') {
             // Для необязательных текстовых полей или если select не select2
            data[key] = null;
        } else {
            data[key] = value;
        }

        // Проверка обязательных полей (особенно select)
        if (element.hasAttribute('required') && (data[key] === null || data[key] === '')) {
            if (element.tagName === 'SELECT' && data[key] === '') { // Пустое значение для select
                 hasEmptyRequiredSelect = true;
                 console.warn(`Обязательное поле-select '${key}' не заполнено.`);
            } else if (data[key] === null) {
                 console.warn(`Обязательное поле '${key}' не заполнено.`);
                 // Можно добавить визуальную обратную связь для пользователя
            }
        }
    });

    if (hasEmptyRequiredSelect) {
        alert("Please fill in all required fields, including dropdowns.");
        // Можно подсветить незаполненные поля
        return; // Остановить отправку
    }
    

    // Приведение типов для числовых полей
    ['movie_year', 'budget'].forEach(key => {
        if (data[key] !== null && data[key] !== undefined && data[key] !== '') {
            data[key] = parseFloat(data[key]);
            if (isNaN(data[key])) data[key] = null; // Если не удалось преобразовать в число
        } else {
            data[key] = null; // Если пусто или null, оставляем null
        }
    });
    
    // Поля, которые могут быть пустыми (и должны отправляться как null, если пусты)
    // и не являются числовыми, уже обработаны выше (formData.forEach)

    const resultContainer = document.getElementById('result-container');
    const resultP = document.getElementById('prediction-result');
    resultP.textContent = 'Predicting...';
    resultContainer.style.display = 'block'; // Показываем контейнер с результатом

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });

        const responseBody = await response.json(); // Читаем тело ответа в любом случае

        if (!response.ok) {
            throw new Error(responseBody.error || `HTTP error! status: ${response.status}`);
        }
        
        // Форматируем число с запятыми как разделителями тысяч
        const predictionValue = parseFloat(responseBody.prediction);
        resultP.textContent = isNaN(predictionValue) ? "Invalid prediction" : `$${predictionValue.toLocaleString(undefined, {minimumFractionDigits: 0, maximumFractionDigits: 0})}`;

    } catch (error) {
        console.error('Error during prediction:', error);
        resultP.textContent = `Error: ${error.message}`;
    }
});