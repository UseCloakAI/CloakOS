import Cocoa
import FlutterMacOS

// Borderless-looking window: content runs under a transparent titlebar, traffic lights stay.
class MainFlutterWindow: NSWindow {
  override func awakeFromNib() {
    let controller = FlutterViewController()
    let frame = self.frame
    self.contentViewController = controller
    self.setFrame(frame, display: true)
    self.titlebarAppearsTransparent = true
    self.titleVisibility = .hidden
    self.styleMask.insert(.fullSizeContentView)
    self.isMovableByWindowBackground = true
    self.minSize = NSSize(width: 900, height: 600)
    self.setContentSize(NSSize(width: 1280, height: 800))
    self.center()
    RegisterGeneratedPlugins(registry: controller)
    super.awakeFromNib()
  }
}
