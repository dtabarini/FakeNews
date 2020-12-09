$(document).ready((_) => {
    $('#submit').click(e => {
        var title = `${$('#title').val()}`.trim();
        var text = `${$('#text').val()}`.trim();
        var model = `${$('#model').val()}`.trim();
        if (title != '' && text != '' && model != '') {
            $('#output').html(
                `<img id="loading" src="img/loading.svg" />`
            );
            setTimeout(_ => {
                $.ajax({
                    url: 'model',
                    method: 'post',
                    dataType: 'json',
                    data: {
                        title: title,
                        text: text,
                        model: model
                    },
                    error: (xhr, status, error) => {
                        console.log(status);
                        console.log(xhr.responseJSON);
                        console.error(error);
                        $('#output').html(status).css('color', '#F34336');;
                    },
                    success: (response, status, xhr) => {
                        console.log(status);
                        console.log(response);
                        if (response) {
                            if (response.status) {
                                var output = (`${response.data.prediction}`).toUpperCase();
                                console.log(output);
                                if (output == 'REAL') {
                                    $('#output')
                                        .html(output)
                                        .css('color', '#06E576');
                                } else {
                                    $('#output')
                                        .html(output)
                                        .css('color', '#F34336');
                                }
                            } else {
                                $('#output').html(
                                    `error: ${response.data.message}`
                                ).css('color', '#F34336');;
                            }
                        } else $('#output').html('no response').css('color', '#F34336');;
                    },
                });
            }, 500);
        }
    });
});