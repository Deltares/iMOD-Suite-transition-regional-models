# Build self-extractable environment yourself

Make sure ``pixi-pack`` and ``pixi-unpack`` are installed globally:

```powershell
pixi global install pixi-pack pixi-unpack
```

Initially run this command once:

```powershell
pixi run clone
```

Consequently to pack the pixi environment into a file, run:

```powershell
pixi run pack
```

This will generate a ``environment.ps1`` file. This contains the zipped pixi
environment. See the [Pixi Pack docs for more info](https://pixi.prefix.dev/latest/deployment/pixi_pack/).
