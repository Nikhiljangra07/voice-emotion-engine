// ear_multiout.swift — create a Multi-Output Device (speakers + BlackHole)
// and set it as the system default output. Replaces the Audio MIDI Setup
// GUI step for the live-ear loopback capture.
//
// Build/run:  swiftc scripts/ear_multiout.swift -o out/ear_multiout && out/ear_multiout
// Revert:     out/ear_multiout revert   (default output back to speakers)

import CoreAudio
import Foundation

func deviceList() -> [AudioDeviceID] {
    var addr = AudioObjectPropertyAddress(
        mSelector: kAudioHardwarePropertyDevices,
        mScope: kAudioObjectPropertyScopeGlobal,
        mElement: kAudioObjectPropertyElementMain)
    var size: UInt32 = 0
    AudioObjectGetPropertyDataSize(AudioObjectID(kAudioObjectSystemObject),
                                   &addr, 0, nil, &size)
    var ids = [AudioDeviceID](repeating: 0,
                              count: Int(size) / MemoryLayout<AudioDeviceID>.size)
    AudioObjectGetPropertyData(AudioObjectID(kAudioObjectSystemObject),
                               &addr, 0, nil, &size, &ids)
    return ids
}

func strProp(_ id: AudioDeviceID, _ sel: AudioObjectPropertySelector) -> String {
    var addr = AudioObjectPropertyAddress(
        mSelector: sel, mScope: kAudioObjectPropertyScopeGlobal,
        mElement: kAudioObjectPropertyElementMain)
    var cf: Unmanaged<CFString>?
    var size = UInt32(MemoryLayout<Unmanaged<CFString>?>.size)
    let st = withUnsafeMutablePointer(to: &cf) {
        AudioObjectGetPropertyData(id, &addr, 0, nil, &size, $0)
    }
    if st != noErr { return "" }
    return cf?.takeRetainedValue() as String? ?? ""
}

func findDevice(named: String) -> (AudioDeviceID, String)? {
    for id in deviceList() {
        let name = strProp(id, kAudioObjectPropertyName)
        if name == named {
            return (id, strProp(id, kAudioDevicePropertyDeviceUID))
        }
    }
    return nil
}

func setDefaultOutput(_ id: AudioDeviceID) {
    var addr = AudioObjectPropertyAddress(
        mSelector: kAudioHardwarePropertyDefaultOutputDevice,
        mScope: kAudioObjectPropertyScopeGlobal,
        mElement: kAudioObjectPropertyElementMain)
    var dev = id
    let st = AudioObjectSetPropertyData(
        AudioObjectID(kAudioObjectSystemObject), &addr, 0, nil,
        UInt32(MemoryLayout<AudioDeviceID>.size), &dev)
    print(st == noErr ? "default output set." : "ERROR setting default: \(st)")
}

let AGG_UID = "com.voiceemotion.ear.multiout"

if CommandLine.arguments.contains("revert") {
    guard let (spk, _) = findDevice(named: "MacBook Pro Speakers") else {
        print("speakers not found"); exit(1)
    }
    setDefaultOutput(spk)
    // destroy the aggregate if present
    for id in deviceList()
    where strProp(id, kAudioDevicePropertyDeviceUID) == AGG_UID {
        AudioHardwareDestroyAggregateDevice(id)
        print("multi-output device removed.")
    }
    print("reverted to MacBook Pro Speakers.")
    exit(0)
}

guard let (_, spkUID) = findDevice(named: "MacBook Pro Speakers") else {
    print("ERROR: MacBook Pro Speakers not found"); exit(1)
}
guard let (_, bhUID) = findDevice(named: "BlackHole 2ch") else {
    print("ERROR: BlackHole 2ch not found (driver loaded?)"); exit(1)
}
print("speakers UID: \(spkUID)")
print("blackhole UID: \(bhUID)")

// reuse if it already exists
var aggID: AudioDeviceID = 0
for id in deviceList()
where strProp(id, kAudioDevicePropertyDeviceUID) == AGG_UID {
    aggID = id
    print("multi-output device already exists (id \(id)) — reusing.")
}

if aggID == 0 {
    let desc: [String: Any] = [
        kAudioAggregateDeviceNameKey as String: "Ear Multi-Output",
        kAudioAggregateDeviceUIDKey as String: AGG_UID,
        kAudioAggregateDeviceIsStackedKey as String: 1,
        kAudioAggregateDeviceMainSubDeviceKey as String: spkUID,
        kAudioAggregateDeviceSubDeviceListKey as String: [
            [kAudioSubDeviceUIDKey as String: spkUID],
            [kAudioSubDeviceUIDKey as String: bhUID,
             kAudioSubDeviceDriftCompensationKey as String: 1],
        ],
    ]
    let st = AudioHardwareCreateAggregateDevice(desc as CFDictionary, &aggID)
    guard st == noErr else {
        print("ERROR creating multi-output device: \(st)"); exit(1)
    }
    print("created 'Ear Multi-Output' (id \(aggID)).")
}
setDefaultOutput(aggID)
print("System audio now plays through speakers AND BlackHole.")
print("Capture with:  .venv_diar/bin/python scripts/live_ear.py --device 0")
