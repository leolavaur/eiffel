{

  description = "Evaluation Framework for FL-based intrusion detection using Flower.";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-22.11";

    poetry2nix = {
      url = "github:nix-community/poetry2nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { self, nixpkgs, poetry2nix, ... }:
    let
      eiffelOverlay = (final: prev:
        let
          eiffelConfig = {
            projectDir = self;
            preferWheels = true;
            python = final.${pythonVer};
            extras = if final.stdenv.isDarwin then [ "darwin" ] else [ "linux" ];
            overrides = prev.poetry2nix.defaultPoetryOverrides.extend (self: super: {
              tensorflow-io-gcs-filesystem = super.tensorflow-io-gcs-filesystem.overrideAttrs (old: {
                buildInputs = old.buildInputs ++ [ prev.libtensorflow ];
              });
            });
          };
        in
        {

          eiffel = final.poetry2nix.mkPoetryApplication eiffelConfig;

          eiffelEnv = final.poetry2nix.mkPoetryEnv eiffelConfig // {
            editablePackageSources = { eiffel = ./.; };
          };

          poetry = (prev.poetry.override { python = final.${pythonVer}; });
        });

      forEachSystem = systems: func: nixpkgs.lib.genAttrs systems (system:
        func (import nixpkgs {
          inherit system;
          config.allowUnfree = true;
          overlays = [
            poetry2nix.overlay
            eiffelOverlay
          ];
        })
      );

      forAllSystems = func: (forEachSystem [ "x86_64-linux" "aarch64-darwin" ] func);

      pythonVer =
        let
          versionList = builtins.filter builtins.isString
            (builtins.split ''\.''
              (builtins.elemAt
                (builtins.match ''^[>=~^]*([0-9]+(\.[0-9]+)*)(,[0-9<=.]*)?$''
                  (builtins.fromTOML (builtins.readFile ./pyproject.toml)).tool.poetry.dependencies.python
                ) 0
              )
            );
        in
        "python${builtins.elemAt versionList 0}${builtins.elemAt versionList 1}";

    in
    {
      devShells = forAllSystems (pkgs: with pkgs; {
        default = mkShellNoCC {
          packages = [
            # this package            
            eiffelEnv

            # development dependencies
            poetry
          ];

          shellHook = ''
            export PYTHONPATH=${pkgs.${pythonVer}}
          '' + (if stdenv.isLinux then ''
            export LD_LIBRARY_PATH=${ lib.strings.concatStringsSep ":" [
              "${cudaPackages.cudatoolkit}/lib"
              "${cudaPackages.cudatoolkit.lib}/lib"
              "${cudaPackages.cudnn}/lib"
            ]}
          '' else "") + ":$LD_LIBRARY_PATH";
        };

      });

      packages = forAllSystems (pkgs: {
        default = pkgs.eiffel;

        poetry = pkgs.poetry;
      });

    };
}
