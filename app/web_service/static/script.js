// static/script.js
$(document).ready(function() {
    // Инициализация статических Select2 (например, MPAA)
    $('.select2-static').select2({
        allowClear: true,
        width: '100%',
        placeholder: $(this).data('placeholder') || 'Select an option'
    });

    // Инициализация AJAX Select2
    $('.select2-ajax').each(function() {
        const fieldName = $(this).attr('id'); // Используем ID для определения типа данных
        const isRequired = $(this).prop('required'); // Проверяем, является ли поле обязательным

        $(this).select2({
            ajax: {
                url: '/get_options', // Эндпоинт во Flask для получения опций
                dataType: 'json',
                delay: 250, // Задержка перед отправкой запроса после ввода
                data: function (params) {
                    return {
                        q: params.term, // Поисковый запрос пользователя
                        field: fieldName, // Сообщаем серверу, для какого поля ищем
                        page: params.page || 1
                    };
                },
                processResults: function (data, params) {
                    params.page = params.page || 1;
                    return {
                        results: data.items, // Ожидаем {id: 'value', text: 'label'}
                        pagination: {
                            more: data.pagination.more // Для бесконечной прокрутки
                        }
                    };
                },
                cache: true
            },
            placeholder: $(this).data('placeholder') || 'Search or select an option',
            allowClear: !isRequired, // Разрешить очистку только если поле не обязательное
            minimumInputLength: 0, // Можно начать поиск сразу или после N символов (например, 1 или 2)
            width: '100%'
        });
    });

    // Обработчик отправки формы
    $('#prediction-form').on('submit', async function(event) {
        event.preventDefault();

        const formData = new FormData(this);
        const data = {};
        let formIsValid = true;

        formData.forEach((value, key) => {
            const element = this.elements[key];
            let processedValue = value;

            // Для Select2, если значение пустое или не выбрано, оно может быть null или ""
            if ($(element).hasClass('select2-ajax') || $(element).hasClass('select2-static')) {
                processedValue = $(element).val(); // Получаем значение из Select2
                if (processedValue === null || processedValue === "") {
                    processedValue = null;
                }
            } else if (element.type === 'number' && value === '') {
                processedValue = null;
            } else if (value === '') {
                processedValue = null;
            }
            data[key] = processedValue;

            // Проверка обязательных полей
            if (element.hasAttribute('required') && (data[key] === null || data[key] === '')) {
                formIsValid = false;
                // Можно добавить класс ошибки к полю element или его родительскому .form-group
                $(element).closest('.form-group').addClass('error-field');
                console.warn(`Обязательное поле '${key}' не заполнено.`);
            } else {
                 $(element).closest('.form-group').removeClass('error-field');
            }
        });

        if (!formIsValid) {
            alert("Please fill in all required fields.");
            return;
        }
        
        // Приведение типов для числовых полей
        ['movie_year', 'budget'].forEach(key => {
            if (data[key] !== null && data[key] !== undefined) {
                data[key] = parseFloat(data[key]);
                if (isNaN(data[key])) data[key] = null;
            } else {
                data[key] = null;
            }
        });

        const resultContainer = $('#result-container');
        const resultP = $('#prediction-result');
        resultP.text('Predicting...');
        resultContainer.show();

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data)
            });

            const responseBody = await response.json();

            if (!response.ok) {
                throw new Error(responseBody.error || `HTTP error! status: ${response.status}`);
            }
            
            const predictionValue = parseFloat(responseBody.prediction);
            resultP.text(isNaN(predictionValue) ? "Invalid prediction" : `$${predictionValue.toLocaleString(undefined, {minimumFractionDigits: 0, maximumFractionDigits: 0})}`);

        } catch (error) {
            console.error('Error during prediction:', error);
            resultP.text(`Error: ${error.message}`);
        }
    });
});