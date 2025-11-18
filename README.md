# Docker Workspace Template

## How to install

First, add all necessary repositories to the [`project.repos`](project.repos) file following the provided example format.

Then run the installation script and select your desired project type:

```bash
./install.sh
```

To cleanup your workspace and remove setup files, run:

```bash
./clean_setup_files.sh
```

This will remove the entire `setup/` folder as well as the installation script.

### Dependencies

The following tools are required to use this Docker workspace template:

- [VS Code](https://code.visualstudio.com/) - Primary development environment
- [Git](https://git-scm.com/) - Version control system
- [Docker](https://docs.docker.com/get-docker/) with [NVIDIA Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) - Container platform with GPU support
- [Python 3](https://www.python.org/downloads/) - Required for setup scripts
- [Python venv](https://docs.python.org/3/library/venv.html) - Virtual environment support

## How to use

This Docker workspace template provides a streamlined development environment with two main components:

- **Base Dockerfile**: Contains the pre-built foundation with common dependencies and tools
- **User Dockerfile**: Your customizable layer where you add project-specific requirements

> [!CAUTION]
> This tool runs all containers with `--privileged` flag for maximum compatibility!
> If you don't need privileged access, modify the `docker/run.sh` script accordingly for better security.

### Project Management

#### Using VS Code Tasks

The most convenient way to manage your project is through the **VS Code task buttons** located at the bottom of your editor. These provide one-click access to common operations like:

- Building Docker images
- Running containers
- Connecting to running containers
- Cleaning up Docker resources

#### Manual Configuration

For advanced users or automated workflows, you can **configure settings manually** by editing the `project.env` file. This file contains environment variables that control:

- Docker image names and tags
- Repository URLs and branches
- Build and runtime parameters
- Custom environment variables for your project

#### Project Structure

Your **project repositories will be automatically cloned** into the `project/` folder based on the repositories listed in `project.repos`. This project folder is **mounted as a volume** inside the Docker container, ensuring:

- Real-time synchronization between your host and container
- Persistent data that survives container restarts
- Seamless development workflow with your favorite IDE

## How to personalize

## How to contribute

## Open ToDo's

- [ ] TEST!
- [ ] Add support for git svn (currently only branches are allowed as versions)
- [ ] Make Dockerhub options more modular
- [ ] Implement dynamic README generation based on project type
    - [X] Basic stitching
    - [ ] Create Versions for each Projecttype
- [X] Create base image for PyTorch from source compilation
- [ ] Add standalone Python installation option
- [ ] Add standalone CUDA installation option
