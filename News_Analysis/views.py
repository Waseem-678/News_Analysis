from django.shortcuts import render, redirect
# from django.views.generic.edit import FormView
# from .forms import FileFieldForm
import asyncio
from .processing import image_processing, text_files_processing
import time
from .spark_init import spark, Row
from .classification import model
from .models import ImageAnalysis, TextFilesAnalysis


def home(request):
    return render(request, 'home.html')


def images(request):
    if request.method == 'POST':
        start_time = time.time()
        temp = request.FILES.getlist('images')
        # loop = asyncio.get_event_loop()
        # main_ = asyncio.wait(*[processing(i) for i in temp], return_when=asyncio.FIRST_COMPLETED)
        # results = loop.run_until_complete(main_)
        filename = []
        for i in temp:
            filename.append(i.name)
            image_processing(i, spark, model, Row)
        # loop.close()
        print("--- %s seconds ---" % (time.time() - start_time))
        results = ImageAnalysis.objects.filter(name__in=filename)
        return render(request, 'News_Analysis/image_details.html', {'results': results})
    return render(request, 'News_Analysis/images.html')


def text_files(request):
    if request.method == 'POST':
        temp = request.FILES.getlist('textfiles')
        start_time = time.time()
        files = []
        filename = []
        for i in temp:
            filename.append(i.name)
            files.append(text_files_processing(i, spark, model, Row))
        print("--- %s seconds ---" % (time.time() - start_time))
        # predictions = []
        # objs = []
        # for i in filename:
        #     objs.append(TextFilesAnalysis.objects.filter(i))
        #
        # for i in objs:
        #     predictions.append(i.prediction)
        results = TextFilesAnalysis.objects.filter(name__in=filename)
        # context = {'files': files, 'results': results}
        return render(request, 'News_Analysis/text_files_details.html', {'results': results})
    return render(request, 'News_Analysis/text_files.html')


# def details(request):
#     # category_id = request.session.bar
#     # image = Image.objects.all()
#     # for i in image:
#     #     img = cv2.imread(i.image.url)
#     #     print(ocr(img))   , {'images': image}
#     return render(request, 'News_Analysis/text_files_details.html')

# class ImageUploadView(View):
#     def get(self, request):
#         # photos_list = Image.objects.all()   , {'photos': photos_list}
#         return render(self.request, 'uploading/text_files_details.html')
#
#     def post(self, request):
#         form = ImageForm(self.request.POST, self.request.FILES)
#         file = request.FILES.get("images", None)
#         print(file)
#         if form.is_valid():
#             photo = form.save()
#             data = {'is_valid': True, 'name': photo.file.name, 'url': photo.file.url}
#         else:
#             data = {'is_valid': False}
#         return JsonResponse(data)
#
#

# class FileFieldView(FormView):
#     form_class = FileFieldForm
#     template_name = 'home.html'  # Replace with your template.
#     success_url = 'details/'  # Replace with your URL or reverse().
#
#     def get(self, request, *args, **kwargs):
#         form = self.form_class()
#         template = self.template_name
#         return render(request, template, {'form': form})
#
#     def post(self, request, *args, **kwargs):
#         form_class = self.get_form_class()
#         form = self.get_form(form_class)
#         files = request.FILES.getlist('file_field')
#         if form.is_valid():
#             form = form(request.POST, request.FILES.getlist('file_field'))
#             return redirect(self.success_url)
#         else:
#             return self.form_invalid(form)



# text = []
#         temp = request.FILES.getlist('images')
#         start_time = time.time()
#         for item in temp:
#             img = cv2.imdecode(numpy.fromstring(item.file.getvalue(), numpy.uint8), cv2.IMREAD_COLOR)
#             text.append(ocr(img))
#         for f in temp:
#             instance = Image(image=f)  # match the model.
#             instance.save()
#         for t in text:
#             print(t)
#         print("--- %s seconds ---" % (time.time() - start_time))
#         return redirect('details/')
#     return render(request, 'uploading/images.html')

