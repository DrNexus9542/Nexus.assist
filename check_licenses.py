import importlib.metadata

print(f"{'Paquet':<25} | {'Version':<10} | {'Licence'}")
print("-" * 55)

for dist in importlib.metadata.distributions():
    name = dist.metadata['Name']
    version = dist.metadata['Version']

    # Chercher la licence dans les métadonnées
    license_info = dist.metadata.get('License', '')

    # Si vide, chercher dans les classificateurs
    if not license_info:
        classifiers = dist.metadata.get_all('Classifier', [])
        for c in classifiers:
            if c.startswith('License ::'):
                license_info = c.split('::')[-1].strip()
                break

    if not license_info:
        license_info = "Non spécifiée"

    print(f"{name:<25} | {version:<10} | {license_info}")