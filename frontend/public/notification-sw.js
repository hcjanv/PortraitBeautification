self.addEventListener("notificationclick", (event) => {
  event.notification.close();

  const data = event.notification.data || {};
  const targetUrl = data.url || "/";
  const messageType = data.messageType || "portrait-beautification:show-result";

  event.waitUntil(
    (async () => {
      const windows = await self.clients.matchAll({
        type: "window",
        includeUncontrolled: true,
      });

      for (const client of windows) {
        const clientUrl = new URL(client.url);
        const target = new URL(targetUrl, self.location.origin);
        if (clientUrl.origin === target.origin) {
          await client.focus();
          client.postMessage({ type: messageType });
          return;
        }
      }

      await self.clients.openWindow(targetUrl);
    })(),
  );
});
