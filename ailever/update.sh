tar -zcf analysis.tar.gz analysis && mv analysis.tar.gz ../storage/
tar -zcf apps.tar.gz apps && mv apps.tar.gz ../storage/
tar -zcf captioning.tar.gz captioning && mv captioning.tar.gz ../storage/
tar -zcf detection.tar.gz detection && mv detection.tar.gz ../storage/
tar -zcf forecast.tar.gz forecast && mv forecast.tar.gz ../storage/
tar -zcf language.tar.gz language && mv language.tar.gz ../storage/
tar -zcf utils.tar.gz utils && mv utils.tar.gz ../storage/

zip -r analysis.zip analysis && mv analysis.zip ../storage/
zip -r apps.zip apps && mv apps.zip ../storage/
zip -r captioning.zip captioning && mv captioning.zip ../storage/
zip -r detection.zip detection && mv detection.zip ../storage/
zip -r forecast.zip forecast && mv forecast.zip ../storage/
zip -r language.zip language && mv language.zip ../storage/
zip -r utils.zip utils && mv utils.zip ../storage/

