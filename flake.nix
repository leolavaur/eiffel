{

  description = "Evaluation Framework for FL-based intrusion detection using Flower.";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-22.11";

    flake-utils.url = "github:numtide/flake-utils";

    p2nflake = {
      url = "github:nix-community/poetry2nix";
      inputs.nixpkgs.follows = "nixpkgs";

    };
  };

  outputs = { self, nixpkgs, flake-utils, p2nflake, ... }:

    flake-utils.lib.eachDefaultSystem (system:
      let

        pkgs = import nixpkgs { 
          inherit system;
          config.allowUnfree = true;
          overlays = [
            p2nflake.overlay # s.default
            eiffelOverlay
          ];
        };

        poetryPath = if pkgs.stdenv.isDarwin then ./macos else ./.;

        # Automatically select the correct Python version from pyproject.toml.
        pythonVer =
          let
            pyproject = builtins.fromTOML (builtins.readFile (poetryPath + /pyproject.toml));
            compatiblePythons = builtins.match
              ''^[>=~^]*([0-9]+(\.[0-9]+)*)(,[0-9<=.]*)?$''
              pyproject.tool.poetry.dependencies.python;
            firstVersion = builtins.elemAt compatiblePythons 0;
            versionAsList = builtins.filter builtins.isString
              (builtins.split ''\.'' firstVersion);
            verstionMajor = builtins.elemAt versionAsList 0;
            verstionMinor = builtins.elemAt versionAsList 1;
          in "python${verstionMajor}${verstionMinor}";

        eiffelOverlay = (final: prev:
          let
            eiffelConfig = {
              projectDir = ./.;
              pyproject = poetryPath + /pyproject.toml;
              poetrylock = poetryPath + /poetry.lock;
              preferWheels = true;
              python = final.${pythonVer};
              overrides = final.poetry2nix.defaultPoetryOverrides.extend (self: super: {
                tensorflow-io-gcs-filesystem = super.tensorflow-io-gcs-filesystem.overrideAttrs (old: {
                  buildInputs = old.buildInputs ++ [ pkgs.libtensorflow ];
                });
                gpustat = super.gpustat.overrideAttrs (old: {
                  buildInputs = (old.buildInputs or [ ]) ++ [ super.setuptools super.setuptools-scm ];
                });
                pandarallel = super.pandarallel.overrideAttrs (old: {
                  buildInputs = (old.buildInputs or [ ]) ++ [ super.setuptools ];
                });
                opencensus = super.opencensus.overrideAttrs (old: {
                  # See: https://github.com/DavHau/mach-nix/issues/255#issuecomment-812984772
                  postInstall = ''
                    rm $out/lib/python3.10/site-packages/opencensus/common/__pycache__/__init__.cpython-310.pyc
                    rm $out/lib/python3.10/site-packages/opencensus/__pycache__/__init__.cpython-310.pyc
                  '';
                });
                ml-dtypes = super.ml-dtypes.overrideAttrs (old: {
                  buildInputs = (old.buildInputs or [ ]) ++ [ super.setuptools super.pybind11 ];
                });
                tensorboard = super.tensorboard.overrideAttrs (old: {
                  buildInputs = (old.buildInputs or [ ]) ++ [ super.six ];
                });
              });
            };
          in
          {

            eiffel = final.poetry2nix.mkPoetryApplication (eiffelConfig // 
                (final.lib.attrsets.optionalAttrs (final.stdenv.isLinux) {
                  LD_LIBRARY_PATH = final.lib.strings.concatStringsSep ":" (with final; [
                    "${cudaPackages.cudatoolkit}/lib"
                    "${cudaPackages.cudatoolkit.lib}/lib"
                    "${cudaPackages.cudnn}/lib"
                    "${cudaPackages.cudatoolkit}/nvvm/libdevice/"
                  ]);
                })
              );

            eiffel-env = final.poetry2nix.mkPoetryEnv (eiffelConfig // {
              editablePackageSources = { eiffel = ./.; };
            });

          }
        );

      in {
        
        devShells = {
        
          default = with pkgs; mkShellNoCC {
            packages = [
              # this package            
              eiffel-env

              # development dependencies
              poetry
            ];

            shellHook = ''
              export EIFFEL_PYTHON_PATH=${eiffel-env}/bin/python
            '' + (if stdenv.isLinux then ''
              export LD_LIBRARY_PATH=${ lib.strings.concatStringsSep ":" [
                  "${cudaPackages.cudatoolkit}/lib"
                  "${cudaPackages.cudatoolkit.lib}/lib"
                  "${cudaPackages.cudnn}/lib"
                  "${cudaPackages.cudatoolkit}/nvvm/libdevice/"
                ] }
              
              export XLA_FLAGS=--xla_gpu_cuda_data_dir=${cudaPackages.cudatoolkit}
            '' else "");
          };

        };

        packages = rec {
          eiffel = pkgs.eiffel;
          eiffel-env = pkgs.eiffel-env;
          default = eiffel;
        };

        overlays = {
          eiffel = eiffelOverlay;
        };

      }
    );
}
