## Installation

`~/.profile`
```bash
# SDK Man replaced GVM. Using for Groovy, Gradle, and Maven Version Management
export SDKMAN_DIR="$HOME/.sdkman"
[[ -s "$HOME/.sdkman/bin/sdkman-init.sh" ]] && source "$HOME/.sdkman/bin/sdkman-init.sh"
```

```bash
$ sudo apt update
$ sudo apt install zip
$ curl -s https://get.sdkman.io | bash
$ sdk install kotlin
```


```bash
$ kotlinc-jvm
```


### Compile
```bash
$ kotlinc [file_name.kt] -include-runtime -d [file_name.jar]
```

### Execution
```bash
$ java -jar [file_name.jar]
```
