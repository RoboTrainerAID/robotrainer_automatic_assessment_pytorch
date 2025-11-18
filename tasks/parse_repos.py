import yaml

if __name__ == "__main__":
    with open("project.repos", "r") as f:
        data = yaml.safe_load(f)

    folders = []
    urls = []
    versions = []

    repos = data.get("repositories", {}) or {}

    for folder, meta in repos.items():
        folders.append(folder)
        urls.append(meta.get("url", ""))
        versions.append(meta.get("version", "main"))

    for i, (folder, url, version) in enumerate(zip(folders, urls, versions)):
        print(f'FOLDER_{i}="{folder}" URL_{i}="{url}" VERSION_{i}="{version}"')
