import traceback

try:
    import app.main
    print("Import successful")
except Exception as e:
    print("Import failed:")
    traceback.print_exc()
