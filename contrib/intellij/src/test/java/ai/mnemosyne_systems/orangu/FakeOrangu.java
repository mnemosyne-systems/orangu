// Copyright (C) 2026 The orangu community
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

package ai.mnemosyne_systems.orangu;

import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Base64;
import java.util.List;

/**
 * A stand-in for orangu in the tests, in plain Java so it runs with nothing
 * but the JDK. With {@code -p}, what it does depends on the prompt:
 *
 * <ul>
 *   <li>{@code fail}: a note on stderr and exit code 3</li>
 *   <li>{@code stream}: the answer in pieces, with escape sequences cut between them</li>
 *   <li>{@code sleep}: starts a child that sleeps too, and never ends by itself</li>
 *   <li>anything else: reports what it was given, one line each: the arguments
 *       (each in base64, so quotes and new lines can be checked exactly), the
 *       folder, NO_COLOR and whether stdin is closed</li>
 * </ul>
 */
public final class FakeOrangu {
    private FakeOrangu() {
    }

    public static void main(String[] args) throws Exception {
        PrintStream out = new PrintStream(System.out, true, StandardCharsets.UTF_8);
        PrintStream err = new PrintStream(System.err, true, StandardCharsets.UTF_8);
        if (args.length == 1 && args[0].equals("--sleep")) {
            Thread.sleep(Long.MAX_VALUE);
            return;
        }
        String prompt = null;
        for (int i = 0; i + 1 < args.length; i++) {
            if (args[i].equals("-p")) {
                prompt = args[i + 1];
            }
        }
        if (prompt == null) {
            out.println("orangu stand-in interface in " + Paths.get("").toAbsolutePath());
            Thread.sleep(Long.MAX_VALUE);
            return;
        }
        // A note, in colour, the way a tool run might print one.
        err.print("\u001b[2mUsing the stand-in server\u001b[0m\n");
        err.flush();
        switch (prompt) {
            case "fail":
                err.print("error: the stand-in server is not running\n");
                err.flush();
                System.exit(3);
                break;
            case "stream":
                for (String piece : new String[] {"Hello", ", \u001b[1", "mworld\u001b[0m", "!\r\n"}) {
                    out.print(piece);
                    out.flush();
                    Thread.sleep(50);
                }
                break;
            case "sleep": {
                String java = Paths.get(System.getProperty("java.home"), "bin", "java").toString();
                Process child = new ProcessBuilder(
                        java, "-cp", System.getProperty("java.class.path"), FakeOrangu.class.getName(), "--sleep")
                        .inheritIO()
                        .start();
                err.print("sleeping, child " + child.pid() + "\n");
                err.flush();
                Thread.sleep(Long.MAX_VALUE);
                break;
            }
            default: {
                boolean[] closed = {false};
                Thread reader = new Thread(() -> {
                    try {
                        closed[0] = System.in.read() < 0;
                    } catch (Exception e) {
                        closed[0] = true;
                    }
                });
                reader.setDaemon(true);
                reader.start();
                reader.join(1000);
                List<String> encoded = new ArrayList<>();
                for (String arg : args) {
                    encoded.add(Base64.getEncoder().encodeToString(arg.getBytes(StandardCharsets.UTF_8)));
                }
                out.print("args " + String.join(" ", encoded) + "\n");
                out.print("cwd " + Paths.get("").toAbsolutePath() + "\n");
                String noColor = System.getenv("NO_COLOR");
                out.print("noColor " + (noColor == null ? "-" : noColor) + "\n");
                out.print("stdinClosed " + closed[0] + "\n");
                out.flush();
                err.print("Done: 1 request\n");
                err.flush();
                System.exit(0);
            }
        }
    }
}
