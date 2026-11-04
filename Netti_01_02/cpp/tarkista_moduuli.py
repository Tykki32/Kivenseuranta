"""CMake-kaannon jalkitarkistus (CMakeLists.txt, POST_BUILD): latautuuko laajennusmoduuli (argumentti, oletus netti_fit) taman Pythonin kanssa. Tulostaa Pythonin version ja
tiedoston paatteen seka, jos lataus epaonnistuu, syyn. Poistuu aina koodilla 0 (ei kaada kaannosta)."""
import importlib
import os
import sys
import sysconfig

nimi = sys.argv[1] if len(sys.argv) > 1 else "netti_fit"
sys.path.insert(0, os.getcwd())                 # skriptin oma kansio (cpp/) olisi muuten ainoa polku; moduuli on projektikansiossa (tyohakemisto)
print(f"  Kaannoksen Python {sys.version.split()[0]} ({sys.executable}), moduulin paate {sysconfig.get_config_var('EXT_SUFFIX')}")
try:
    m = importlib.import_module(nimi)
    print(f"  {nimi} latautuu: {m.versio() if hasattr(m, 'versio') else 'ok'}")
    print("  Aja ohjelmat TALLA Pythonilla (muuten moduulia ei loydy eika kaytetty).")
except Exception as e:
    print(f"  VAROITUS: {nimi} ei lataudu taman Pythonin kanssa: {e!r}")
    print("  ImportError / DLL load failed: OpenCV:n DLL-tiedostot puuttuvat moduulin vieresta (vcpkg kopioi ne yleensa automaattisesti).")
sys.exit(0)
