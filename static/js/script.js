// Front-end validation and form handling
document.addEventListener('DOMContentLoaded', function() {
    const form = document.querySelector('form');

    if (form) {
        form.addEventListener('submit', function(event) {
            const imageInput = document.querySelector('input[name="image"]');
            const hemoglobinInput = document.querySelector('input[name="hemoglobin"]');
            const mchInput = document.querySelector('input[name="mch"]');
            const mchcInput = document.querySelector('input[name="mchc"]');
            const mcvInput = document.querySelector('input[name="mcv"]');
            const tabularDataInput = document.querySelector('input[name="tabular_data"]');

            // Validate image is selected
            if (!imageInput.files || imageInput.files.length === 0) {
                event.preventDefault();
                alert('Please select an image file');
                return;
            }

            // Validate all blood parameters are entered and are valid numbers
            if (!isValidNumber(hemoglobinInput.value)) {
                event.preventDefault();
                alert('Please enter a valid Hemoglobin value');
                return;
            }

            if (!isValidNumber(mchInput.value)) {
                event.preventDefault();
                alert('Please enter a valid MCH value');
                return;
            }

            if (!isValidNumber(mchcInput.value)) {
                event.preventDefault();
                alert('Please enter a valid MCHC value');
                return;
            }

            if (!isValidNumber(mcvInput.value)) {
                event.preventDefault();
                alert('Please enter a valid MCV value');
                return;
            }

            // Combine the individual values into the tabular_data field for backward compatibility
            const combinedValues = [
                parseFloat(hemoglobinInput.value),
                parseFloat(mcvInput.value),
                parseFloat(mchInput.value),
                parseFloat(mchcInput.value)
            ].join(',');

            tabularDataInput.value = combinedValues;
        });
    }

    // Preview image when selected
    const imageInput = document.querySelector('input[name="image"]');
    if (imageInput) {
        imageInput.addEventListener('change', function() {
            // You could add code here to preview the selected image if desired
        });
    }

    // Function to validate number inputs
    function isValidNumber(value) {
        if (value === '' || value === null) return false;
        const num = parseFloat(value);
        return !isNaN(num) && num >= 0;
    }
});